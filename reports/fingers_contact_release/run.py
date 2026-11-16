"""FingersSphere2d ablations with the original cost and interpolation start."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujodico import mjx,trajopt
from mujodico.examples import methods
from mujodico.examples.tasks import FingersSphere2d
from mujodico.mjx._src import solver_sap
from mujodico.mjx._src import solver_sap_util,math as mjx_math
from mujodico.trajopt._src.transcriptions import _put_optimization_model,_extract_contact_indices

jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'


def encode(v):
  if isinstance(v,bytes):return v.decode()
  if isinstance(v,(np.ndarray,jax.Array)):return np.asarray(v).tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)


def save_json(path,value):path.write_text(json.dumps(value,default=encode,indent=2))


def release_events(gaps,contact_threshold=.0005,release_threshold=.002,min_release=3):
  events=[]
  had_contact=False;release_start=None
  for k,gap in enumerate(gaps):
    if gap<=contact_threshold:
      if release_start is not None and k-release_start>=min_release:
        events.append({'release':release_start,'recontact':k,'duration_steps':k-release_start})
      had_contact=True;release_start=None
    elif had_contact and gap>=release_threshold and release_start is None:
      release_start=k
  return events


def main():
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--label',required=True)
  ap.add_argument('--form',choices=['forward','implicit','shooting'],default='forward')
  ap.add_argument('--solver',choices=['ipopt','ippas'],default='ipopt')
  ap.add_argument('--force-floor',type=float,default=0.,help='Force lower bound is minus this value; inf removes it.')
  ap.add_argument('--force-bound-kind',choices=['edges','normal'],default='edges')
  ap.add_argument('--gap-margin',type=float,help='Add physical contact-distance lower bounds of minus this margin.')
  ap.add_argument('--gap-copies',type=int,default=4,help='Gap inequalities per contact, matching the four pyramid edges by default.')
  ap.add_argument('--gap-kind',choices=['position','predicted'],default='position')
  ap.add_argument('--smoothing',type=float,default=1e-5)
  ap.add_argument('--hessian',choices=['gn','full','bfgs'],default='gn')
  ap.add_argument('--iterations',type=int,default=500)
  ap.add_argument('--mu-init',type=float,default=.1)
  ap.add_argument('--mu-strategy',choices=['monotone','adaptive'],default='monotone')
  ap.add_argument('--inner-tolerance',type=float)
  ap.add_argument('--inner-iterations',type=int)
  ap.add_argument('--init',help='Previous label or path to .npz for an optimizer continuation.')
  ap.add_argument('--warm-duals',action='store_true')
  ap.add_argument('--force-scale',type=float,default=1.)
  ap.add_argument('--geometry-width',type=float,default=0.,help='C2 smoothing width for capsule projection, as fraction of segment length.')
  ap.add_argument('--relu',choices=['reciprocal','algebraic_matched'],default='reciprocal')
  ap.add_argument('--c2-velocity',action='store_true')
  ap.add_argument('--tau-factor',type=float,help='Explicit HC extra time scale divided by dt; zero matches the zero-smoothing physical limit.')
  ap.add_argument('--accept-every',action='store_true')
  ap.add_argument('--print-level',type=int,default=0)
  ap.add_argument('--max-wall-time',type=float,default=1200.)
  args=ap.parse_args()
  DATA.mkdir(parents=True,exist_ok=True)
  original_projection=mjx_math.closest_segment_point
  original_relu=solver_sap_util.reciprocal_approx_relu
  original_relu_grad=solver_sap_util.reciprocal_approx_relu_grad
  original_local=solver_sap_util.local_smooth_relu
  original_local_grad=solver_sap_util.local_smooth_relu_grad
  original_hc=solver_sap_util.hunt_crossley_normal_impulse
  original_hc_grad=solver_sap_util.hunt_crossley_normal_impulse_grad
  def c2_relu(x,e):
    s=jp.clip((x+e)/(2*e),0.,1.)
    return jp.where(x<=-e,0.,jp.where(x>=e,x,e*(2*s**3-s**4)))
  def c2_grad(x,e):
    s=jp.clip((x+e)/(2*e),0.,1.)
    return jp.where(x<=-e,0.,jp.where(x>=e,1.,3*s*s-2*s**3))
  if args.geometry_width>0:
    def projection(a,b,pt):
      ab=b-a;t=jp.dot(pt-a,ab)/(jp.dot(ab,ab)+1e-6)
      e=args.geometry_width
      return a+(c2_relu(t,e)-c2_relu(t-1,e))*ab
    mjx_math.closest_segment_point=projection
  if args.relu=='algebraic_matched':
    def algebraic(x,e):
      r=jp.sqrt(x*x+e*e)
      return jp.where(x<0,e*e/(2*(r-jp.minimum(x,0.))),.5*(x+r))
    def algebraic_grad(x,e):
      r=jp.sqrt(x*x+e*e)
      return jp.where(x<0,e*e/(2*r*(r-jp.minimum(x,0.))),.5*(1+x/r))
    solver_sap_util.reciprocal_approx_relu=algebraic
    solver_sap_util.reciprocal_approx_relu_grad=algebraic_grad
  if args.c2_velocity:
    solver_sap_util.local_smooth_relu=c2_relu
    solver_sap_util.local_smooth_relu_grad=c2_grad
  if args.tau_factor is not None:
    def hc(vn,phi0,dt,k,d,contact_smoothing,bounce_speed_tol,next_step=True):
      t=dt*(1+args.tau_factor) if next_step else dt*args.tau_factor
      x1=-phi0-t*vn;x2=1/d-vn
      # The explicit hard branch avoids differentiating sqrt(0). It has the
      # original zero-smoothing values and the correct derivative off the kink.
      if args.smoothing==0.:
        penetration=jp.maximum(x1,0.)
      else:
        eps=jp.sqrt(4*contact_smoothing*t/(k*dt))
        penetration=solver_sap_util.reciprocal_approx_relu(x1,eps)
      return dt*k*d*penetration*solver_sap_util.local_smooth_relu(x2,bounce_speed_tol)
    def hc_grad(vn,phi0,dt,k,d,contact_smoothing,bounce_speed_tol,next_step=True):
      t=dt*(1+args.tau_factor) if next_step else dt*args.tau_factor
      x1=-phi0-t*vn;x2=1/d-vn
      if args.smoothing==0.:
        penetration=jp.maximum(x1,0.)
        slope=jp.where(x1>0,1.,jp.where(x1<0,0.,.5))
      else:
        eps=jp.sqrt(4*contact_smoothing*t/(k*dt))
        penetration=solver_sap_util.reciprocal_approx_relu(x1,eps)
        slope=solver_sap_util.reciprocal_approx_relu_grad(x1,eps)
      return -dt*k*d*(t*slope*solver_sap_util.local_smooth_relu(x2,bounce_speed_tol)
                     +penetration*solver_sap_util.local_smooth_relu_grad(x2,bounce_speed_tol))
    solver_sap_util.hunt_crossley_normal_impulse=hc
    solver_sap_util.hunt_crossley_normal_impulse_grad=hc_grad
  task=FingersSphere2d()
  if args.inner_tolerance is not None:
    task.model.opt.tolerance=task.model.opt.ls_tolerance=args.inner_tolerance
  if args.inner_iterations is not None:
    task.model.opt.iterations=task.model.opt.ls_iterations=args.inner_iterations
  object.__setattr__(task,'contact_smoothing',args.smoothing*1000)
  started=time.perf_counter()
  print('Building',args.label,vars(args),flush=True)
  trans=methods.LiftedMultipleShootingTranscription(task)
  model=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=args.smoothing)
  data=mjx.make_data(model)
  addresses,contacts=_extract_contact_indices(model)
  nc=0 if args.form=='shooting' else len(addresses)
  lb,ub=map(np.copy,task.u_bounds if args.form=='shooting' else trans.w_bounds)
  lb[task.nu:]=-args.force_floor if args.force_bound_kind=='edges' else -np.inf
  nw=len(lb)

  def dynamics(x,w):
    if args.form!='shooting':return trans.dynamics(x,w)
    d=data.replace(qpos=x[:model.nq],qvel=x[model.nq:model.nq+model.nv],ctrl=w)
    d=mjx.step(model,d)
    return jp.concatenate([d.qpos,d.qvel,d.act])

  def stage_data(x,u):
    d=data.replace(qpos=x[:model.nq],qvel=x[model.nq:model.nq+model.nv],ctrl=u)
    d=mjx.fwd_position(model,d)
    return d

  def constitutive(x,w,xn):
    if args.form=='shooting':return jp.zeros((0,))
    if args.form=='forward':return trans.stage_equality(x,w)
    d=stage_data(x,w[:task.nu])
    d=mjx.fwd_velocity(model,d)
    d=mjx.fwd_actuation(model,d)
    d=mjx.fwd_acceleration(model,d)
    ctx=solver_sap.Context.create(model,d)
    ctx=ctx.replace(qvel=xn[model.nq:model.nq+model.nv])
    force=solver_sap._calc_constraint_impulse(ctx)[addresses]/model.opt.timestep
    return w[task.nu:]-force

  def eq(x,w,xn,k,p):
    return jp.concatenate([xn-dynamics(x,w),constitutive(x,w,xn)/args.force_scale])

  def ineq(x,w,xn,k,p):
    result=[]
    if args.form!='shooting' and args.force_bound_kind=='normal' and np.isfinite(args.force_floor):
      result.append(jp.stack([jp.sum(w[task.nu+c.cf_index:task.nu+c.cf_index+4])+args.force_floor for c in contacts]))
    if args.gap_margin is not None:
      d=stage_data(x,w[:task.nu])
      if args.gap_kind=='predicted':
        result.append(d._impl.efc_pos[addresses]+2*model.opt.timestep*(d._impl.efc_J[addresses]@xn[model.nq:model.nq+model.nv])+args.gap_margin)
      else:result.append(jp.repeat(d._impl.contact.dist+args.gap_margin,args.gap_copies))
    return jp.concatenate(result) if result else jp.zeros((0,))

  history=[]
  problem=None
  best_feasible=np.inf
  def callback(stats,p):
    nonlocal best_feasible
    h={k:v for k,v in stats.items() if k!='x'}
    v=problem.get_current_violations(scaled=False)
    h['unscaled']={k:float(np.max(np.abs(a),initial=0)) for k,a in v.items()}
    history.append(h)
    if h['unscaled']['g_violation']<1e-5 and h['objective']<best_feasible-1e-4:
      best_feasible=h['objective']
      current=problem.get_current_iterate(scaled=False)
      cx,cw=nlp._unpack(stats['x'])
      checkpoint_dir=DATA/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
      checkpoint=checkpoint_dir/(args.label+'_best')
      np.savez_compressed(checkpoint.with_suffix('.npz'),xs=cx,ws=cw)
      save_json(checkpoint.with_suffix('.json'),{'args':vars(args),'iteration':stats['iter'],
          'initial_state_control_hash':common_hash,'info':{
          'obj_val':h['objective'],'mult_g':current['mult_g'],
          'mult_x_L':current['mult_x_L'],'mult_x_U':current['mult_x_U']}})
    if h['iter']%50==0:print(json.dumps(h,default=encode),flush=True)
    return p

  solver=trajopt.IpoptTrajOpt(N=trans.N,nx=trans.nx,nu=nw,
      local_cost=lambda x,w,xn,k,p:trans.stage_cost(x,w,p),terminal_cost=trans.terminal_cost,
      initial_equality_constraint=lambda x,p:x-p.x_0,local_equality_constraint=eq,
      local_inequality_constraint=ineq,x_bounds=trans.x_bounds,u_bounds=(lb,ub),
      gauss_newton=args.hessian!='full',params_trace=trans.params,update_params=callback)
  nlp=solver._nlp
  xs0=task.xs_init;ws0=np.hstack([task.us_init,np.zeros((trans.N,nc))])
  original_hash=hashlib.sha256(xs0.tobytes()+ws0.tobytes()).hexdigest()
  common_hash=hashlib.sha256(xs0.tobytes()+task.us_init.tobytes()).hexdigest()
  previous=None
  if args.init:
    init_path=Path(args.init)
    if init_path.suffix!='.npz':init_path=DATA/(args.init+'.npz')
    old=np.load(init_path);xs0=old['xs'];ws0=old['ws']
    if args.form=='shooting':ws0=ws0[:,:task.nu]
    if ws0.shape[1]!=nw:raise ValueError('Continuation input dimension changed; use matching contact sets and formulations.')
    previous=json.loads(init_path.with_suffix('.json').read_text())
  z0=nlp._pack(xs0,ws0)
  problem=nlp.to_cyipopt()
  options=dict(max_iter=args.iterations,max_wall_time=args.max_wall_time,print_level=args.print_level,mu_init=args.mu_init,
               mu_strategy=args.mu_strategy,accept_every_trial_step='yes' if args.accept_every else 'no')
  if args.hessian=='bfgs':options['hessian_approximation']='limited-memory'
  kwargs={}
  if args.warm_duals:
    if previous is None:raise ValueError('--warm-duals requires --init')
    if len(previous['info']['mult_g'])!=nlp.num_constraints:raise ValueError('Constraint dimensions changed')
    options.update(warm_start_init_point='yes',warm_start_bound_push=1e-8,warm_start_bound_frac=1e-8,
                   warm_start_mult_bound_push=1e-12)
    kwargs=dict(lagrange=np.array(previous['info']['mult_g']),zl=np.array(previous['info']['mult_x_L']),
                zu=np.array(previous['info']['mult_x_U']))
  for k,v in options.items():problem.add_option(k,v)
  compile_seconds=time.perf_counter()-started
  started=time.perf_counter()
  if args.solver=='ipopt':
    z,info=problem.solve(z0,**kwargs)
  else:
    if args.warm_duals:raise ValueError('Dual continuation currently supports IPOPT only')
    ocp=trajopt.IppasTrajOpt(N=trans.N,nx=trans.nx,nu=nw,
      initial_state=lambda p:p.x_0,dynamics=lambda x,w,k,p:dynamics(x,w),
      stage_cost=lambda x,w,k,p:trans.stage_cost(x,w,p),terminal_cost=trans.terminal_cost,
      stage_equality_constraint=lambda x,w,k,p:constitutive(x,w,dynamics(x,w))/args.force_scale,
      stage_inequality_constraint=lambda x,w,k,p:ineq(x,w,dynamics(x,w),k,p),
      x_bounds=trans.x_bounds,u_bounds=(lb,ub),gauss_newton=args.hessian!='full',params_trace=trans.params)
    ox,ow,info=jax.jit(lambda x,w:ocp.solve(trans.params,x,w,max_iter=args.iterations,mu_init=args.mu_init))(
      jp.asarray(xs0),jp.asarray(ws0))
    jax.block_until_ready((ox,ow,info));info.pop('warm_start',None)
    z=nlp._pack(np.asarray(ox),np.asarray(ow))
    history.append({'iter':info['iter_count'],'mu':info['mu'],'inf_pr':info['primal_error'],
                    'inf_du':info['dual_error'],'objective':info['obj_val']})
  solve_seconds=time.perf_counter()-started
  xs,ws=nlp._unpack(z);xs,ws=np.asarray(xs),np.asarray(ws)
  c=np.asarray(nlp.constraints(z));cl,cu=nlp.constraint_bounds
  violation=np.maximum(np.maximum(cl-c,c-cu),0)
  local=c[trans.nx:].reshape(trans.N,-1)
  rawforce=np.asarray(jax.jit(jax.vmap(lambda x,w,xn:constitutive(x,w,xn)))(
      jp.asarray(xs[:-1]),jp.asarray(ws),jp.asarray(xs[1:])))

  # All validation uses the original capsule geometry and reciprocal HC law.
  mjx_math.closest_segment_point=original_projection
  solver_sap_util.reciprocal_approx_relu=original_relu
  solver_sap_util.reciprocal_approx_relu_grad=original_relu_grad
  solver_sap_util.local_smooth_relu=original_local
  solver_sap_util.local_smooth_relu_grad=original_local_grad
  solver_sap_util.hunt_crossley_normal_impulse=original_hc
  solver_sap_util.hunt_crossley_normal_impulse_grad=original_hc_grad

  # Evaluate geometry independent of the force decisions, including terminal state.
  @jax.jit
  def geometry(x):
    d=stage_data(x,jp.zeros(task.nu))
    return d._impl.contact.dist,d._impl.contact.geom,d._impl.contact.pos
  gaps,geoms,points=[np.asarray(a) for a in jax.jit(jax.vmap(geometry))(jp.asarray(xs))]
  geoms=geoms[0]
  # The colliding sphere is the first geom on the object body; its indicator is disabled.
  object_body=mujoco.mj_name2id(task.model,mujoco.mjtObj.mjOBJ_BODY,'object')
  object_geom=int(task.model.body_geomadr[object_body])
  finger_ids={side:[i for i in range(task.model.ngeom)
                   if mujoco.mj_id2name(task.model,mujoco.mjtObj.mjOBJ_BODY,int(task.model.geom_bodyid[i])).startswith(side)]
              for side in ['left','right']}
  masks={side:np.array([object_geom in pair and any(i in pair for i in ids) for pair in geoms])
         for side,ids in finger_ids.items()}

  def contact_stats(gs):
    result={}
    for side,mask in masks.items():
      g=np.min(gs[:,mask],axis=1)
      result[side]={'min_gap':float(np.min(g)),'max_gap':float(np.max(g)),
                    'regrasp_events':release_events(g),
                    'steps_in_contact_0p5mm':int(np.sum(g<=.0005)),
                    'steps_released_2mm':int(np.sum(g>=.002))}
    return result

  # Nominal smooth and zero-smoothing replays, with accurate inner solves.
  replays={}
  replay_metrics={}
  for smoothing,name in [(args.smoothing,'smooth'),(0.,'zero')]:
    replay_task=FingersSphere2d()
    replay_task.model.opt.tolerance=replay_task.model.opt.ls_tolerance=1e-12
    replay_task.model.opt.iterations=replay_task.model.opt.ls_iterations=100
    rm=_put_optimization_model(replay_task.model,contact_approximation=task.contact_approx,contact_smoothing=smoothing)
    rd=mjx.make_data(rm)
    @jax.jit
    def replay(us):
      def step(x,u):
        d=rd.replace(qpos=x[:rm.nq],qvel=x[rm.nq:rm.nq+rm.nv],ctrl=u)
        d=mjx.step(rm,d)
        xn=jp.concatenate([d.qpos,d.qvel,d.act])
        return xn,xn
      _,tail=jax.lax.scan(step,jp.asarray(task.x_0),us)
      return jp.concatenate([jp.asarray(task.x_0)[None],tail])
    rx=np.asarray(replay(jp.asarray(ws[:,:task.nu])))
    rg=np.asarray(jax.jit(jax.vmap(lambda x:geometry(x)[0]))(jp.asarray(rx)))
    replays[name+'_xs']=rx;replays[name+'_gaps']=rg
    cost=sum(float(trans.stage_cost(x,w,trans.params)) for x,w in zip(rx[:-1],ws))
    cost+=float(trans.terminal_cost(rx[-1],trans.params))
    replay_metrics[name]={'objective':cost,'position_error_m':float(np.linalg.norm(rx[-1,4:6]-task.x_goal[4:6])),
       'angle_error_radians':float(abs(rx[-1,6]-task.x_goal[6])),
       'max_state_difference':float(np.max(np.abs(rx-xs))),'contacts':contact_stats(rg)}
  info.pop('x',None)
  xl,xu=nlp.var_bounds
  summary={'args':vars(args),'options':options,'info':info,'initial_linear_hash':original_hash,
     'initial_state_control_hash':common_hash,
     'compile_seconds':compile_seconds,'solve_seconds':solve_seconds,
     'objective':float(nlp.objective(z)),'initial_objective':float(nlp.objective(z0)),
     'status':int(info['status']),'iterations':history[-1]['iter'],
     'last_iteration':history[-1],'ncf':nc,'nx':trans.nx,'N':trans.N,
     'initial_state':task.x_0,'goal_state':task.x_goal,
     'constraint_violation':float(np.max(violation)),'force_equality_defect':float(np.max(np.abs(rawforce),initial=0)),
     'dynamics_defect':float(np.max(np.abs(local[:,:trans.nx]))),
     'bound_violation':float(np.max(np.maximum(np.maximum(xl-z,z-xu),0))),
     'position_error_m':float(np.linalg.norm(xs[-1,4:6]-task.x_goal[4:6])),
     'angle_error_radians':float(abs(xs[-1,6]-task.x_goal[6])),
     'contacts':contact_stats(gaps),'replay':replay_metrics,
     'contact_pairs':geoms,'object_geom':object_geom,
     'finger_contact_masks':masks}
  save_json(DATA/(args.label+'.json'),summary);save_json(DATA/(args.label+'_history.json'),history)
  np.savez_compressed(DATA/(args.label+'.npz'),xs=xs,ws=ws,gaps=gaps,points=points,**replays)
  print('RESULT',args.label,{k:summary[k] for k in ['status','objective','position_error_m','angle_error_radians',
        'constraint_violation','force_equality_defect','contacts','replay']},flush=True)


if __name__=='__main__':main()
