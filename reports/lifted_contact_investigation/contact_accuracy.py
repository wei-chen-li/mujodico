"""Compare default/tight contact solves at fixed outer-NLP iterates."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import numpy as np
from mujodico import mjx
from mujodico.examples import tasks,methods
from mujodico.mjx._src import solver_sap
from mujodico.trajopt._src.transcriptions import _put_optimization_model
from convergence_diagnostics import ROOT,OUT,dump

jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')


def probe(label):
  saved=json.loads((ROOT/'data'/(label+'.json')).read_text())
  raw=np.load(ROOT/'data'/(label+'.npz'))
  kkt=json.loads((OUT/(label+'_kkt.json')).read_text())
  a=saved['args']
  collected=[]
  for tight in [False,True]:
    task=getattr(tasks,a['task'])(a['seed'])
    if tight:
      task.model.opt.tolerance=task.model.opt.ls_tolerance=1e-12
      task.model.opt.iterations=task.model.opt.ls_iterations=100
    object.__setattr__(task,'contact_smoothing',a['smoothing']*1000)
    trans=methods.LiftedMultipleShootingTranscription(task)
    ncf=trans.nw-task.nu
    force=lambda x,u:-trans.stage_equality(x,jp.concatenate([u,jp.zeros(ncf)]))
    values=np.asarray(jax.jit(jax.vmap(force))(jp.asarray(raw['xs'][:-1]),jp.asarray(raw['ws'][:,:task.nu])))
    k=kkt['top_force_defects'][0]['stage']
    v=jp.asarray(np.concatenate([raw['xs'][k],raw['ws'][k,:task.nu]]))
    f=lambda v:force(v[:trans.nx],v[trans.nx:])
    J=np.asarray(jax.jit(jax.jacfwd(f))(v))
    fd=[]
    for h in [1e-6,1e-8]:
      D=np.asarray(jax.jit(jax.vmap(lambda d:(f(v+d)-f(v-d))/(2*h)))(jp.eye(len(v))*h)).T
      fd.append({'step':h,'max_absolute_error':float(np.max(np.abs(D-J))),
                 'relative_row_error':float(np.max(np.linalg.norm(D-J,axis=1)/np.maximum(1,np.linalg.norm(J,axis=1))))})
    model=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=a['smoothing'])
    data=mjx.make_data(model)
    @jax.jit
    def inner_residual(x,u):
      d=mjx.forward(model,data.replace(qpos=x[:model.nq],qvel=x[model.nq:model.nq+model.nv],ctrl=u))
      context=solver_sap.Context.create(model,d)
      context=context.replace(qvel=d.qvel+model.opt.timestep*d.qacc)
      g=solver_sap._calc_cost_gradient(context)
      delta=solver_sap._calc_step_direction(context).delta_qvel
      return jp.max(jp.abs(g)),jp.max(jp.abs(delta))
    gr,dv=[np.asarray(v) for v in jax.jit(jax.vmap(inner_residual))(
        jp.asarray(raw['xs'][:-1]),jp.asarray(raw['ws'][:,:task.nu]))]
    collected.append({'tight':tight,'forces':values,'jacobian_check_stage':k,'finite_difference_checks':fd,
      'max_inner_momentum_residual':float(np.max(gr)),
      'max_inner_newton_velocity_correction':float(np.max(dv)),
      'inner_newton_velocity_correction_at_worst_force_stage':float(dv[k]),
      'default_tolerance':float(task.model.opt.tolerance)})
  difference=np.abs(collected[0]['forces']-collected[1]['forces'])
  index=np.unravel_index(np.argmax(difference),difference.shape)
  result={'label':label,'evaluations':collected,'max_force_change_tightening_inner':float(np.max(difference)),
     'max_force_change_stage':int(index[0]),'max_force_change_edge':int(index[1]),
     'default_force_at_max_change':collected[0]['forces'][index],
     'tight_force_at_max_change':collected[1]['forces'][index],
     'force_equality_residual_tight_max':float(np.max(np.abs(raw['ws'][:,task.nu:]-collected[1]['forces'])))}
  dump(OUT/(label+'_inner_accuracy.json'),result)
  print(label,{k:result[k] for k in ['max_force_change_tightening_inner','force_equality_residual_tight_max']},
        [(d['tight'],d['max_inner_newton_velocity_correction'],d['finite_difference_checks']) for d in collected],flush=True)


if __name__=='__main__':
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--labels',nargs='+',default=['sphere_baseline','box_baseline','circular_baseline','square_baseline'])
  args=ap.parse_args()
  for label in args.labels:probe(label)
