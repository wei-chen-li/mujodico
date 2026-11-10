"""Compare IP-PAS and CasADi-FATROP iteration traces on a lifted contact OCP."""
import sys, os, time, numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import casadi as ca
from jax_casadi import JaxCallback
from cmp_lib import compare
from mujodico.examples.tasks import parse_task_name
from mujodico.examples.methods import IppasLiftedMultipleShooting as M, util
from mujodico.trajopt._src.fatrop_ref.ippas import IpPasSolver, IpPasOptions
task_name=sys.argv[1]; seed=int(sys.argv[2]); MI=int(sys.argv[3]); watchdog=int(sys.argv[4]) if len(sys.argv)>4 else 1
task=parse_task_name(task_name)(seed)
opts={} if watchdog else {'watchdog_shortened_iter_trigger':0}
m=M(task, max_iter=MI, record_trace=True, **opts)
sol=m.solver; tr=m.transcription
N,nx,nw,ne,ni=sol.N,sol.nx,sol.nw,sol.ne,sol.ni
print(f'{task_name} seed {seed}: N={N} nx={nx} nw={nw} ne={ne} ni={ni}')
params=m._params()
P=params['p']
f_cb=JaxCallback('f', lambda x,w: sol.dynamics(x,w), [nx,nw], nx)
h_cb=JaxCallback('h', lambda x,w: sol.stage_equality(x,w), [nx,nw], ne) if ne else None
g_cb=JaxCallback('g', lambda x,w: sol.stage_inequality(x,w), [nx,nw], ni)
Q=np.asarray(P.Q); Qf=np.asarray(P.Qf); R=np.asarray(P.R); xg=np.asarray(P.x_goal); ur=np.asarray(P.u_ref); nu=task.nu
assert not task.quaternion_w_idxs
X=[ca.MX.sym(f'x{k}',nx) for k in range(N+1)]; W=[ca.MX.sym(f'w{k}',nw) for k in range(N)]
v=[];g=[];lbg=[];ubg=[];eqf=[];J=0
for k in range(N):
    v+=[X[k],W[k]]
    if k==0: g.append(X[0]-ca.DM(np.asarray(P.x_0))); lbg+=[0]*nx; ubg+=[0]*nx; eqf+=[True]*nx
    g.append(X[k+1]-f_cb(X[k],W[k])); lbg+=[0]*nx; ubg+=[0]*nx; eqf+=[True]*nx
    if ne: g.append(h_cb(X[k],W[k])); lbg+=[0]*ne; ubg+=[0]*ne; eqf+=[True]*ne
    g.append(g_cb(X[k],W[k])); lbg+=[0]*ni; ubg+=[np.inf]*ni; eqf+=[False]*ni
    e=X[k]-xg; du=W[k][:nu]-ur
    J+=ca.mtimes([e.T,Q,e])+ca.mtimes([du.T,R,du])
v.append(X[N]); e=X[N]-xg; J+=ca.mtimes([e.T,Qf,e])
o={'structure_detection':'auto','equality':eqf,'fatrop.print_level':10,'fatrop.max_iter':MI}
if not watchdog: o['fatrop.watchdog_shortened_iter_trigger']=0
S=ca.nlpsol('s','fatrop',{'x':ca.vertcat(*v),'f':J,'g':ca.vertcat(*g)},o)
xs0=np.asarray(task.xs_init); ws0=np.concatenate([task.us_init,np.zeros((N,nw-nu))],1)
xinit=np.concatenate([np.concatenate([xs0[k],ws0[k]]) for k in range(N)]+[xs0[N]])
import re
r_,wfd=os.pipe(); saved=os.dup(1); os.dup2(wfd,1)
t0=time.perf_counter()
try: res=S(x0=xinit,lbg=lbg,ubg=ubg)
finally: sys.stdout.flush(); os.dup2(saved,1); os.close(wfd)
tc=time.perf_counter()-t0
txt=os.read(r_,10**8).decode(errors='replace'); os.close(r_)
rows=[]
for line in txt.splitlines():
    mm=re.match(r'\s*(\d+)(r?)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\d]+)\s+([-+.\deE]+)\s+(\S+)\s+([-+.\deE]+)\s+([-+.\deE]+)(\S)\s+(\d+)',line)
    if mm: rows.append(mm.groups())
xv=np.asarray(res['x']).ravel(); xN_c=xv[-nx:]
St=sol.solve(params, xs0, ws0); out=IpPasSolver.summarize(St)
fd,txtc=compare(rows,out['trace'])
print(txtc)
print('casadi status',S.stats()['return_status'],'rows',len(rows),'time %.1fs'%tc,'| ippas',out['status'],out['iterations'],'resto',out['n_restorations'],'| first divergence row',fd)
print('goal err casadi',util.state_cost(task.Q,task.x_goal,xN_c,[]),'ippas',util.state_cost(task.Q,task.x_goal,np.asarray(out['x'][-1]),[]))
