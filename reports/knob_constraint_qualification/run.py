"""Headless knob contact-mode and constraint-qualification experiments."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import numpy as np
from tasks import make_task
from mujodico import mjx,trajopt
from mujodico.examples import methods
from mujodico.trajopt._src.transcriptions import _put_optimization_model,_extract_contact_indices

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data'
jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')

def encode(v):
  if isinstance(v,bytes):return v.decode()
  if isinstance(v,(np.ndarray,jax.Array)):return np.asarray(v).tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)

def dump(p,v):p.write_text(json.dumps(v,default=encode,indent=2))

def build(args,callback=None):
  task=make_task(args.task,tip_only=args.tip_only,friction=args.friction,goal_degrees=args.goal_degrees,
                 finger_collisions=getattr(args,'finger_collisions',False),mirror_axis=getattr(args,'mirror_axis','y'),
                 same_joint_angles=getattr(args,'same_joint_angles',False))
  if args.inner_tolerance is not None:
    task.model.opt.tolerance=task.model.opt.ls_tolerance=args.inner_tolerance
    task.model.opt.iterations=task.model.opt.ls_iterations=100
  object.__setattr__(task,'contact_smoothing',args.smoothing*1000)
  tr=methods.LiftedMultipleShootingTranscription(task)
  model=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=args.smoothing)
  data=mjx.make_data(model)
  if args.form=='shooting':
    nw=task.nu;bounds=task.u_bounds
    def dynamics(x,w):
      d=mjx.step(model,data.replace(qpos=x[:model.nq],qvel=x[model.nq:],ctrl=w))
      return jp.r_[d.qpos,d.qvel,d.act]
    forceeq=lambda x,w:jp.zeros((0,))
  else:
    nw=tr.nw;bounds=tuple(np.array(v) for v in tr.w_bounds)
    bounds[0][task.nu:]=-args.force_floor
    dynamics=tr.dynamics;forceeq=tr.stage_equality
  solver=trajopt.IpoptTrajOpt(N=tr.N,nx=tr.nx,nu=nw,
      local_cost=lambda x,w,xn,k,p:tr.stage_cost(x,w,p),terminal_cost=tr.terminal_cost,
      initial_equality_constraint=lambda x,p:x-p.x_0,
      local_equality_constraint=lambda x,w,xn,k,p:jp.r_[xn-dynamics(x,w),forceeq(x,w)],
      x_bounds=tr.x_bounds,u_bounds=bounds,gauss_newton=args.hessian!='full',
      params_trace=tr.params,update_params=callback)
  return task,tr,solver,model,data,nw,dynamics,forceeq,bounds

def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--label',required=True)
  ap.add_argument('--task',choices=['push','quadrant','finger','two'],default='finger')
  ap.add_argument('--tip-only',action='store_true');ap.add_argument('--friction',type=float)
  ap.add_argument('--finger-collisions',action='store_true')
  ap.add_argument('--mirror-axis',choices=['x','y'],default='y')
  ap.add_argument('--same-joint-angles',action='store_true')
  ap.add_argument('--goal-degrees',type=float,default=360.)
  ap.add_argument('--force-floor',type=float,default=0.)
  ap.add_argument('--smoothing',type=float,default=1e-5)
  ap.add_argument('--form',choices=['lifted','shooting'],default='lifted')
  ap.add_argument('--solver',choices=['ipopt','ippas'],default='ipopt')
  ap.add_argument('--hessian',choices=['gn','full','bfgs'],default='gn')
  ap.add_argument('--inner-tolerance',type=float);ap.add_argument('--iterations',type=int,default=500)
  ap.add_argument('--init');ap.add_argument('--max-wall-time',type=float,default=300.)
  ap.add_argument('--warm-duals',action='store_true')
  ap.add_argument('--mu-init',type=float,default=.1)
  ap.add_argument('--force-init',type=float,default=0.)
  ap.add_argument('--constant-init',action='store_true')
  args=ap.parse_args();DATA.mkdir(exist_ok=True,parents=True)
  problem=None;history=[];snapshots=[];mult=[]
  def callback(h,p):
    row={k:v for k,v in h.items() if k!='x'}
    v=problem.get_current_violations(scaled=False)
    row['unscaled']={k:float(np.max(abs(a),initial=0)) for k,a in v.items()}
    history.append(row)
    if h['iter']%50==0:
      print(args.label,h['iter'],h['objective'],h['inf_pr'],h['inf_du'],flush=True)
      snapshots.append(np.array(h['x']));mult.append(np.array(problem.get_current_iterate(scaled=False)['mult_g']))
    return p
  print('BUILD',vars(args),flush=True)
  task,tr,solver,m,d,nw,dyn,eq,bounds=build(args,callback);nlp=solver._nlp
  x0=task.xs_init;w0=np.hstack([task.us_init,np.full((tr.N,nw-task.nu),args.force_init)])
  if args.constant_init:x0=np.tile(task.x_0,(tr.N+1,1))
  if args.init:
    old=np.load(DATA/(args.init+'.npz'));x0=old['xs'];w0=old['ws'][:,:nw]
  if args.solver=='ipopt':
    problem=nlp.to_cyipopt()
    options=dict(max_iter=args.iterations,max_wall_time=args.max_wall_time,print_level=0,mu_init=args.mu_init)
    if args.hessian=='bfgs':options['hessian_approximation']='limited-memory'
    kwargs={}
    if args.warm_duals:
      source=json.loads((DATA/(args.init+'.json')).read_text())['info']
      options['warm_start_init_point']='yes'
      kwargs=dict(lagrange=np.array(source['mult_g']),zl=np.array(source['mult_x_L']),zu=np.array(source['mult_x_U']))
    for k,v in options.items():problem.add_option(k,v)
    z,info=problem.solve(nlp._pack(x0,w0),**kwargs);xs,ws=nlp._unpack(z)
  else:
    opt=trajopt.IppasTrajOpt(N=tr.N,nx=tr.nx,nu=nw,
        initial_state=lambda p:p.x_0,dynamics=lambda x,w,k,p:dyn(x,w),
        stage_cost=lambda x,w,k,p:tr.stage_cost(x,w,p),terminal_cost=tr.terminal_cost,
        stage_equality_constraint=lambda x,w,k,p:eq(x,w),x_bounds=tr.x_bounds,u_bounds=bounds,
        params_trace=tr.params,gauss_newton=args.hessian!='full')
    xs,ws,info=jax.jit(lambda x,w:opt.solve(tr.params,x,w,max_iter=args.iterations,mu_init=args.mu_init))(x0,w0)
    jax.block_until_ready((xs,ws,info));xs,ws=np.asarray(xs),np.asarray(ws)
    info.pop('warm_start',None);z=nlp._pack(xs,ws)
  info.pop('x',None)
  c=np.asarray(nlp.constraints(z));cl,cu=nlp.constraint_bounds
  @jax.jit
  def geom(x):
    nd=mjx.fwd_position(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:]))
    return nd._impl.contact.dist,nd._impl.contact.geom
  gaps,pairs=map(np.asarray,jax.jit(jax.vmap(geom))(xs));pairs=pairs[0]
  replays={};metrics={};goal_index=m.nq-1
  # Original hard geometry, both native positive smoothing and zero smoothing.
  for name,kappa in [('smooth',args.smoothing),('zero',0.)]:
    task.model.opt.tolerance=task.model.opt.ls_tolerance=1e-12
    task.model.opt.iterations=task.model.opt.ls_iterations=100
    rm=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=kappa)
    rd=mjx.make_data(rm)
    @jax.jit
    def rollout(us):
      def step(x,u):
        nd=mjx.step(rm,rd.replace(qpos=x[:rm.nq],qvel=x[rm.nq:],ctrl=u))
        xn=jp.r_[nd.qpos,nd.qvel,nd.act];return xn,xn
      _,tail=jax.lax.scan(step,jp.asarray(task.x_0),us)
      return jp.concatenate([task.x_0[None],tail])
    rx=np.asarray(rollout(ws[:,:task.nu]));replays[name+'_xs']=rx
    metrics[name]={'angle_degrees':float(np.rad2deg(rx[-1,goal_index])),
      'max_state_difference':float(abs(rx-xs).max()),
      'bound_violation':float(np.maximum(np.maximum(task.x_bounds[0]-rx,rx-task.x_bounds[1]),0).max())}
  summary={'args':vars(args),'info':info,'objective':float(nlp.objective(z)),
    'angle_degrees':float(np.rad2deg(xs[-1,goal_index])),
    'violation':float(np.maximum(np.maximum(cl-c,c-cu),0).max()),
    'nx':tr.nx,'nw':nw,'nu':task.nu,'N':tr.N,'contact_pairs':pairs,
    'replay':metrics,'history':history}
  dump(DATA/(args.label+'.json'),summary)
  np.savez_compressed(DATA/(args.label+'.npz'),xs=xs,ws=ws,gaps=gaps,
      snapshots=np.asarray(snapshots),multipliers=np.asarray(mult),**replays)
  print('RESULT',args.label,summary['angle_degrees'],summary['violation'],info['status'],metrics,flush=True)

if __name__=='__main__':main()
