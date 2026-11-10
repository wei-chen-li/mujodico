import sys, numpy as np, jax, jax.numpy as jnp
from cmp_lib import *
import casadi as ca
N=int(sys.argv[1]); umax=float(sys.argv[2]); hm=sys.argv[3] if len(sys.argv)>3 else 'exact_lagrangian'
nx=2;nw=1;dt=0.1; x0=np.array([0.0,0.0])
def dyn(x,w): return jnp.array([x[0]+dt*x[1], x[1]+dt*(w[0]-9.81*jnp.sin(x[0])-0.1*x[1])])
def dyn_ca(x,w): return ca.vertcat(x[0]+dt*x[1], x[1]+dt*(w[0]-9.81*ca.sin(x[0])-0.1*x[1]))
lb=np.array([-umax,-8.0]); ub=np.array([umax,8.0])
ineq=lambda x,w: jnp.array([w[0],x[1]]); ineq_ca=lambda x,w: ca.vertcat(w[0],x[1])
eq=lambda x,w: jnp.zeros((0,)); eq_ca=lambda x,w: None
cost=lambda x,w,p: (x[0]-np.pi)**2+0.1*x[1]**2+0.01*w[0]**2
cost_ca=lambda x,w: (x[0]-np.pi)**2+0.1*x[1]**2+0.01*w[0]**2
term=lambda x,p: 100*((x[0]-np.pi)**2+x[1]**2); term_ca=lambda x: 100*((x[0]-np.pi)**2+x[1]**2)
xs0=np.zeros((N+1,nx)); xs0[:,0]=np.linspace(0,np.pi,N+1); ws0=np.zeros((N,nw))
MI=200
st,rows,xv,txt=run_casadi(N,nx,nw,x0,dyn_ca,cost_ca,term_ca,eq_ca,ineq_ca,lb,ub,xs0,ws0,MI)
sol=IpPasSolver(horizon=N,nx=nx,nw=nw,stage_cost=cost,terminal_cost=term,dynamics=dyn,stage_equality=eq,stage_inequality=ineq,ineq_lb=lb,ineq_ub=ub,hessian_mode=hm,options=IpPasOptions(max_iter=MI))
S=sol.solve({'x0':jnp.asarray(x0)},xs0,ws0); out=IpPasSolver.summarize(S)
fd,txtc=compare(rows,out['trace'])
print(txtc); print('casadi',st,len(rows),'| ippas',out['status'],out['iterations'],'n_resto',out['n_restorations'],'| first divergence row',fd)
print('xN casadi',xv[-nx:],'ippas',np.asarray(out['x'][-1]))
