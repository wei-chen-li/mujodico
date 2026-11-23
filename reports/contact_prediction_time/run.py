"""Separate Hunt--Crossley prediction time, smoothing width, and derivatives.

All patches are process-local. The production contact implementation is intact.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import numpy as np
from mujodico import mjx,trajopt
from mujodico.examples import methods,tasks
from mujodico.mjx._src import solver_sap_util as su
from mujodico.trajopt._src.transcriptions import _put_optimization_model,_extract_contact_indices

jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'

def encode(v):
  if isinstance(v,bytes):return v.decode()
  if isinstance(v,(np.ndarray,jax.Array)):return np.asarray(v).tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)

def dump(p,v):p.write_text(json.dumps(v,default=encode,indent=2))

def patch(alpha,width=None,manual=None):
  b=alpha if width is None else width
  g=alpha if manual is None else manual
  def value(vn,phi0,dt,k,d,contact_smoothing,bounce_speed_tol,next_step=True):
    t=dt*alpha if next_step else dt*(alpha-1.)
    tw=dt*b if next_step else dt*(b-1.)
    # Restore the original hard-contact law for zero-smoothing replay.
    t=jp.where(contact_smoothing>0,t,dt if next_step else 0.)
    tw=jp.where(contact_smoothing>0,tw,dt if next_step else 0.)
    eps=jp.sqrt(4*contact_smoothing*tw/(k*dt))
    return dt*k*d*su.reciprocal_approx_relu(-phi0-t*vn,eps)*su.local_smooth_relu(1/d-vn,bounce_speed_tol)
  def grad(vn,phi0,dt,k,d,contact_smoothing,bounce_speed_tol,next_step=True):
    t=dt*g if next_step else dt*(g-1.)
    tw=dt*b if next_step else dt*(b-1.)
    t=jp.where(contact_smoothing>0,t,dt if next_step else 0.)
    tw=jp.where(contact_smoothing>0,tw,dt if next_step else 0.)
    eps=jp.sqrt(4*contact_smoothing*tw/(k*dt));x=-phi0-t*vn;y=1/d-vn
    return -dt*k*d*(t*su.reciprocal_approx_relu_grad(x,eps)*su.local_smooth_relu(y,bounce_speed_tol)
       +su.reciprocal_approx_relu(x,eps)*su.local_smooth_relu_grad(y,bounce_speed_tol))
  su.hunt_crossley_normal_impulse=value;su.hunt_crossley_normal_impulse_grad=grad

def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--label',required=True)
  ap.add_argument('--task',default='PushSphere2d');ap.add_argument('--seed',type=int,default=0)
  ap.add_argument('--alpha',type=float,default=2.);ap.add_argument('--width-alpha',type=float)
  ap.add_argument('--manual-alpha',type=float)
  ap.add_argument('--iterations',type=int,default=500);ap.add_argument('--mu-init',type=float,default=.1)
  ap.add_argument('--mu-target',type=float,default=0.);ap.add_argument('--smoothing',type=float,default=2e-5)
  ap.add_argument('--inner-tolerance',type=float);ap.add_argument('--hessian',choices=['gn','full'],default='gn')
  ap.add_argument('--init');ap.add_argument('--solver',choices=['ipopt','ippas'],default='ipopt')
  ap.add_argument('--form',choices=['lifted','reduced'],default='lifted')
  args=ap.parse_args();DATA.mkdir(exist_ok=True,parents=True)
  patch(args.alpha,args.width_alpha,args.manual_alpha)
  task=getattr(tasks,args.task)(args.seed)
  if args.inner_tolerance is not None:
    task.model.opt.tolerance=task.model.opt.ls_tolerance=args.inner_tolerance
    task.model.opt.iterations=task.model.opt.ls_iterations=100
  object.__setattr__(task,'contact_smoothing',args.smoothing*1000)
  tr=methods.LiftedMultipleShootingTranscription(task)
  if args.form=='reduced':
    given_force=tr.dynamics;constitutive=tr.stage_equality;ncf=tr.nw-task.nu
    def force(x,u):return -constitutive(x,jp.r_[u,jp.zeros(ncf)])
    tr.dynamics=lambda x,u:given_force(x,jp.r_[u,force(x,u)])
    tr.stage_equality=lambda x,u:jp.zeros((0,))
    tr.stage_inequality=force;tr.nw=task.nu;tr.w_bounds=task.u_bounds
  m=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=args.smoothing)
  d=mjx.make_data(m);addresses,_=_extract_contact_indices(m)
  hist=[];zs=[];mus=[];problem=None
  def callback(h,p):
    hist.append({k:v for k,v in h.items() if k!='x'})
    zs.append(np.asarray(h['x']).copy())
    mus.append(np.array(problem.get_current_iterate(scaled=False)['mult_g']))
    if h['iter']%50==0:print(args.label,h['iter'],h['objective'],h['inf_pr'],h['mu'],flush=True)
    return p
  opt=trajopt.IpoptTrajOpt(N=tr.N,nx=tr.nx,nu=tr.nw,
    local_cost=lambda x,w,xn,k,p:tr.stage_cost(x,w,p),terminal_cost=tr.terminal_cost,
    initial_equality_constraint=lambda x,p:x-p.x_0,
    local_equality_constraint=lambda x,w,xn,k,p:jp.r_[xn-tr.dynamics(x,w),tr.stage_equality(x,w)],
    local_inequality_constraint=(lambda x,w,xn,k,p:tr.stage_inequality(x,w)) if args.form=='reduced' else None,
    x_bounds=tr.x_bounds,u_bounds=tr.w_bounds,params_trace=tr.params,
    gauss_newton=args.hessian!='full',update_params=callback)
  nlp=opt._nlp
  xs0=task.xs_init;ws0=np.hstack([task.us_init,np.zeros((tr.N,tr.nw-task.nu))])
  if args.init:
    source=Path(args.init)
    if source.suffix!='.npz':source=DATA/(args.init+'.npz')
    raw=np.load(source);xs0,ws0=raw['xs'],raw['ws']
  print('BUILD',vars(args),flush=True)
  if args.solver=='ipopt':
    problem=nlp.to_cyipopt()
    for k,v in dict(max_iter=args.iterations,max_wall_time=300.,print_level=0,
                   mu_init=args.mu_init,mu_target=args.mu_target).items():problem.add_option(k,v)
    z,info=problem.solve(nlp._pack(xs0,ws0));xs,ws=nlp._unpack(z)
  else:
    opt=trajopt.IppasTrajOpt(N=tr.N,nx=tr.nx,nu=tr.nw,
      initial_state=lambda p:p.x_0,dynamics=lambda x,w,k,p:tr.dynamics(x,w),
      stage_cost=lambda x,w,k,p:tr.stage_cost(x,w,p),terminal_cost=tr.terminal_cost,
      stage_equality_constraint=lambda x,w,k,p:tr.stage_equality(x,w),
      stage_inequality_constraint=(lambda x,w,k,p:tr.stage_inequality(x,w)) if args.form=='reduced' else None,
      x_bounds=tr.x_bounds,u_bounds=tr.w_bounds,params_trace=tr.params,gauss_newton=args.hessian!='full')
    xs,ws,info=jax.jit(lambda x,w:opt.solve(tr.params,x,w,max_iter=args.iterations,mu_init=args.mu_init))(xs0,ws0)
    jax.block_until_ready((xs,ws,info));xs,ws=np.asarray(xs),np.asarray(ws)
    info.pop('warm_start',None);z=nlp._pack(xs,ws)
  @jax.jit
  def geometry(x):
    nd=mjx.fwd_position(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:]))
    return nd._impl.contact.dist,nd._impl.efc_J[addresses]
  gap,J=map(np.asarray,jax.jit(jax.vmap(geometry))(xs))
  c=np.asarray(nlp.constraints(z));lo,hi=nlp.constraint_bounds
  local=c[tr.nx:].reshape(tr.N,-1)
  delta=xs[:,:2]-xs[:,2:4];ang=np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
  replays={};metrics={}
  for name,kappa in [('smooth',args.smoothing),('zero',0.)]:
    rm=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=kappa)
    rd=mjx.make_data(rm)
    @jax.jit
    def replay(us):
      def advance(x,u):
        nd=mjx.step(rm,rd.replace(qpos=x[:rm.nq],qvel=x[rm.nq:],ctrl=u))
        xn=jp.r_[nd.qpos,nd.qvel,nd.act];return xn,xn
      _,tail=jax.lax.scan(advance,jp.asarray(task.x_0),us)
      return jp.concatenate([task.x_0[None],tail])
    rx=np.asarray(replay(ws[:,:task.nu]));replays[name+'_xs']=rx
    metrics[name]={'position_error':float(np.linalg.norm(rx[-1,2:4]-task.x_goal[2:4])),
                   'maximum_state_difference':float(abs(rx-xs).max())}
  info.pop('x',None)
  result={'args':vars(args),'info':info,'nx':tr.nx,'nw':tr.nw,'N':tr.N,
    'objective':float(nlp.objective(z)),'goal_error':float(np.linalg.norm(xs[-1,2:4]-task.x_goal[2:4])),
    'angle_span_degrees':float(np.degrees(np.ptp(ang))),'min_gap':float(gap.min()),
    'force_defect':float(abs(local[:,tr.nx:]).max()) if args.form=='lifted' else 0.,
    'dynamics_defect':float(abs(local[:,:tr.nx]).max()),
    'constraint_violation':float(np.maximum(np.maximum(lo-c,c-hi),0).max()),
    'replay':metrics,'history':hist}
  dump(DATA/(args.label+'.json'),result)
  np.savez_compressed(DATA/(args.label+'.npz'),xs=xs,ws=ws,gaps=gap,J=J,
    iterates=np.asarray(zs),multipliers=np.asarray(mus),**replays)
  print('RESULT',args.label,result['goal_error'],result['angle_span_degrees'],info['status'],flush=True)

if __name__=='__main__':main()
