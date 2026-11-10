import sys, numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import casadi as ca
from mujodico.trajopt._src.fatrop_ref.ippas import IpPasSolver, IpPasOptions
N=int(sys.argv[1]) if len(sys.argv)>1 else 10; nx=2; nw=1
x0=np.array([1.0,0.0])
def dyn(x,w): return jnp.array([x[0]+0.1*x[1], x[1]+0.1*w[0]-0.1*jnp.sin(x[0])])
def ineq(x,w): return jnp.array([x[0]+x[1], w[0]])
lb=np.array([-2.0,-0.5]); ub=np.array([np.inf,0.5])
def eq(x,w): return jnp.zeros((0,))
cost=lambda x,w,p: jnp.sum(x**2)+0.1*jnp.sum(w**2)
term=lambda x,p: 10*jnp.sum(x**2)
# casadi
X=[ca.MX.sym(f'x{k}',nx) for k in range(N+1)]; U=[ca.MX.sym(f'u{k}',nw) for k in range(N)]
v=[];g=[];lbg=[];ubg=[];eqf=[];J=0
for k in range(N):
    v+=[X[k],U[k]]
    if k==0: g.append(X[0]-ca.DM(x0)); lbg+=[0,0]; ubg+=[0,0]; eqf+=[True]*2
    xn=ca.vertcat(X[k][0]+0.1*X[k][1], X[k][1]+0.1*U[k][0]-0.1*ca.sin(X[k][0]))
    g.append(X[k+1]-xn); lbg+=[0,0]; ubg+=[0,0]; eqf+=[True]*2
    g.append(ca.vertcat(X[k][0]+X[k][1],U[k][0])); lbg+=list(lb); ubg+=list(ub); eqf+=[False]*2
    J+=ca.sumsqr(X[k])+0.1*ca.sumsqr(U[k])
v.append(X[N]); J+=10*ca.sumsqr(X[N])
S=ca.nlpsol('s','fatrop',{'x':ca.vertcat(*v),'f':J,'g':ca.vertcat(*g)},{'structure_detection':'auto','equality':eqf,'fatrop.print_level':10,'fatrop.max_iter':60})
xs0=np.zeros((N+1,nx)); xs0[0]=x0
r=S(x0=np.concatenate([np.concatenate([xs0[k],[0.0]]) for k in range(N)]+[xs0[N]]),lbg=lbg,ubg=ubg)
print('casadi status',S.stats()['return_status'])
sol=IpPasSolver(horizon=N,nx=nx,nw=nw,stage_cost=cost,terminal_cost=term,dynamics=dyn,stage_equality=eq,stage_inequality=ineq,ineq_lb=lb,ineq_ub=ub,linear_backend=sys.argv[2] if len(sys.argv)>2 else 'pas',hessian_mode=sys.argv[3] if len(sys.argv)>3 else 'exact_lagrangian',options=IpPasOptions(max_iter=60))
St=sol.solve({'x0':jnp.asarray(x0)}, xs0, np.zeros((N,nw)))
out=IpPasSolver.summarize(St)
print('ippas status',out['status'],out['iterations'])
print(IpPasSolver.format_trace(out['trace']))
xc=np.asarray(r['x']).ravel()
print('max |x diff|', np.max(np.abs(xc[-nx:]-np.asarray(out['x'][-1]))))
