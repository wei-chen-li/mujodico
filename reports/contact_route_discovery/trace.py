"""Trace geometric steering and controlled derivative-path/symmetry probes.

Derivative-path ablations keep forward values but deliberately remove selected
geometric derivatives. They diagnose the search, not alternative physical NLPs.
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
from mujodico.trajopt._src.transcriptions import _put_optimization_model,_extract_contact_indices

jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'

def encode(v):
  if isinstance(v,bytes):return v.decode()
  if isinstance(v,(np.ndarray,jax.Array)):return np.asarray(v).tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)

def build(task,mode):
  tr=methods.LiftedMultipleShootingTranscription(task)
  m=_put_optimization_model(task.model,contact_approximation=task.contact_approx,
                             contact_smoothing=task.contact_smoothing*.001)
  d0=mjx.make_data(m);addresses,_=_extract_contact_indices(m)
  original=tr.dynamics
  if mode!='baseline':
    def dynamics(x,w):
      d=d0.replace(qpos=x[:m.nq],qvel=x[m.nq:m.nq+m.nv],ctrl=w[:task.nu])
      d=mjx.fwd_position(m,d);d=mjx.fwd_velocity(m,d)
      d=mjx.fwd_actuation(m,d);d=mjx.fwd_acceleration(m,d)
      J=d._impl.efc_J
      if mode=='no_geometry':J=jax.lax.stop_gradient(J)
      elif mode=='no_normal':J=J.at[:,:4].set(jax.lax.stop_gradient(J[:,:4]))
      elif mode=='no_arm':J=J.at[:,4].set(jax.lax.stop_gradient(J[:,4]))
      force=jp.zeros(d._impl.nefc).at[addresses].set(w[task.nu:])
      d=d.tree_replace({'qfrc_constraint':J.T@force,'_impl.efc_force':force,'qacc':None})
      d=mjx.implicit(m,d)
      return jp.concatenate([d.qpos,d.qvel,d.act])
    tr.dynamics=dynamics
  return tr,m,d0,original

def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--task',default='PushSphere2d')
  ap.add_argument('--label',required=True);ap.add_argument('--iterations',type=int,default=150)
  ap.add_argument('--mode',choices=['baseline','no_geometry','no_normal','no_arm'],default='baseline')
  ap.add_argument('--symmetric-offset',type=float)
  args=ap.parse_args();DATA.mkdir(exist_ok=True,parents=True)
  task=getattr(tasks,args.task)()
  if args.symmetric_offset is not None:
    assert args.task=='PushSphere2d'
    task.x_0[:4]=[-.18,args.symmetric_offset,.3,0.]
  tr,m,d0,original=build(task,args.mode)
  states=[];duals=[];stats=[];problem=None
  def callback(h,p):
    states.append(np.asarray(h['x']).copy())
    duals.append(np.asarray(problem.get_current_iterate(scaled=False)['mult_g']).copy())
    stats.append({k:v for k,v in h.items() if k!='x'})
    if h['iter']%25==0:print(args.label,h['iter'],h['objective'],h['inf_pr'],flush=True)
    return p
  solver=trajopt.IpoptTrajOpt(N=tr.N,nx=tr.nx,nu=tr.nw,
      local_cost=lambda x,w,xn,k,p:tr.stage_cost(x,w,p),terminal_cost=tr.terminal_cost,
      initial_equality_constraint=lambda x,p:x-p.x_0,
      local_equality_constraint=lambda x,w,xn,k,p:jp.concatenate([xn-tr.dynamics(x,w),tr.stage_equality(x,w)]),
      x_bounds=tr.x_bounds,u_bounds=tr.w_bounds,params_trace=tr.params,update_params=callback)
  nlp=solver._nlp;problem=nlp.to_cyipopt()
  for key,value in dict(max_iter=args.iterations,print_level=0,max_wall_time=300.).items():problem.add_option(key,value)
  w0=np.hstack([task.us_init,np.zeros((tr.N,tr.nw-task.nu))])
  z,info=problem.solve(nlp._pack(task.xs_init,w0))
  xs,ws=nlp._unpack(z)
  direction=xs[:,:2]-xs[:,2:4];angle=np.unwrap(np.arctan2(direction[:,1],direction[:,0]))
  # Establish that diagnostic ablations do not change the forward values.
  probe=np.hstack([task.us_init,np.ones((tr.N,tr.nw-task.nu))*.01])
  errors=np.asarray(jax.jit(jax.vmap(lambda x,w:tr.dynamics(x,w)-original(x,w)))(task.xs_init[:-1],probe))
  np.savez_compressed(DATA/(args.label+'.npz'),iterates=np.asarray(states),multipliers=np.asarray(duals),
      xs=xs,ws=ws,x0=task.x_0,goal=task.x_goal)
  info.pop('x',None)
  summary={'args':vars(args),'info':info,'nx':tr.nx,'nw':tr.nw,'N':tr.N,
      'position_error':float(np.linalg.norm(xs[-1,2:4]-task.x_goal[2:4])),
      'angle_span_degrees':float(np.degrees(np.ptp(angle))),
      'max_primal_value_change_from_ablation':float(abs(errors).max()),'history':stats}
  (DATA/(args.label+'.json')).write_text(json.dumps(summary,default=encode,indent=2))
  print('RESULT',args.label,summary['position_error'],summary['angle_span_degrees'],info['status'],flush=True)

if __name__=='__main__':main()
