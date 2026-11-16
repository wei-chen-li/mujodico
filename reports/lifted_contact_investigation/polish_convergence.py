"""Resume a saved moving trajectory and record all IPOPT stopping residuals."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from convergence_diagnostics import ROOT, OUT, build, dump


def main():
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--source',default='sphere_baseline')
  ap.add_argument('--mode',choices=['gn','exact','bfgs'],default='exact')
  ap.add_argument('--iterations',type=int,default=500)
  ap.add_argument('--inner-tolerance',type=float)
  ap.add_argument('--inner-iterations',type=int)
  ap.add_argument('--label')
  ap.add_argument('--checkpoint',help='Resume a polish result stored under convergence/data.')
  ap.add_argument('--trace-stage',type=int,help='Record local box geometry and state stationarity at this stage.')
  args=ap.parse_args()
  label=args.label or args.source+'_polish_'+args.mode
  saved,task,trans,nlp,raw=build(args.source,exact=args.mode=='exact',
      inner_tolerance=args.inner_tolerance,inner_iterations=args.inner_iterations)
  if args.checkpoint:
    saved=json.loads((OUT/(args.checkpoint+'.json')).read_text())
    raw=np.load(OUT/(args.checkpoint+'.npz'))
  start=nlp._pack(raw['xs'],raw['ws'])
  problem=nlp.to_cyipopt()
  history_path=(OUT/(args.checkpoint+'_history.json')) if args.checkpoint else (ROOT/'data'/(args.source+'_history.json'))
  old_history=json.loads(history_path.read_text())
  initial_mu=old_history[-1]['mu']
  options=dict(max_iter=args.iterations, print_level=5, mu_init=initial_mu,
               warm_start_init_point='yes', warm_start_bound_push=1e-9,
               warm_start_bound_frac=1e-9,warm_start_mult_bound_push=1e-12)
  if args.mode=='bfgs':options['hessian_approximation']='limited-memory'
  for k,v in options.items():problem.add_option(k,v)
  history=[]
  final_vectors={}
  def callback(stats,params):
    nonlocal final_vectors
    vectors=problem.get_current_violations(scaled=False)
    scaled=problem.get_current_violations(scaled=True)
    final_vectors=vectors
    h={k:v for k,v in stats.items() if k!='x'}
    h['unscaled_violations']={k:float(np.max(np.abs(v),initial=0)) for k,v in vectors.items()}
    h['scaled_violations']={k:float(np.max(np.abs(v),initial=0)) for k,v in scaled.items()}
    if args.trace_stage is not None:
      states,_=nlp._unpack(stats['x'])
      k=args.trace_stage;x=states[k]
      if task.model.nq==5:rel=x[:2]-x[2:4];theta=x[4];half=np.array([.15,.08])
      else:rel=x[:2];theta=x[2];half=np.array([.08,.08])
      local=np.array([np.cos(theta)*rel[0]+np.sin(theta)*rel[1],-np.sin(theta)*rel[0]+np.cos(theta)*rel[1]])
      h['trace']={'stage':k,'box_local_pusher':local,'projection_margins':np.abs(local)-half,
                  'pusher_y_stationarity':vectors['grad_lag_x'][k*(trans.nx+trans.nw)+1]}
    history.append(h)
    return params
  nlp.update_params=callback
  started=time.perf_counter()
  result,info=problem.solve(start,lagrange=np.asarray(saved['info']['mult_g']),
      zl=np.asarray(saved['info']['mult_x_L']),zu=np.asarray(saved['info']['mult_x_U']))
  seconds=time.perf_counter()-started
  xs,ws=nlp._unpack(result)
  c=np.asarray(nlp.constraints(result))
  local=c[trans.nx:].reshape(trans.N,-1)
  summary={'args':vars(args),'options':options,'info':info,'time_seconds':seconds,
           'last_iteration':history[-1], 'first_iteration':history[0],
           'dynamics_defect_max':float(np.max(np.abs(local[:,:trans.nx]))),
           'force_equality_defect_max':float(np.max(np.abs(local[:,trans.nx:trans.nx+trans.nw-task.nu]))),
           'goal_cost':float((xs[-1]-task.x_goal)@task.Q@(xs[-1]-task.x_goal)),
           'object_position_error_m':None if task.model.nq==3 else float(np.linalg.norm(xs[-1,2:4])),
           'hinge_angle_error_radians':float(abs(xs[-1,2])) if task.model.nq==3 else None}
  summary['info'].pop('x',None)
  dump(OUT/(label+'.json'),summary)
  dump(OUT/(label+'_history.json'),history)
  np.savez_compressed(OUT/(label+'.npz'),xs=xs,ws=ws,**final_vectors)
  print('RESULT',label,info['status'],summary['last_iteration'],flush=True)


if __name__=='__main__':main()
