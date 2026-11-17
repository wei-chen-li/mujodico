"""Replay one-finger controls with the added finger left idle.

This is a feasibility/control-authority diagnostic, not an optimizer warm start.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import numpy as np
from tasks import make_task
from run import ROOT,DATA,dump
from mujodico import mjx
from mujodico.trajopt._src.transcriptions import _put_optimization_model

def main():
  source=np.load(DATA/'finger.npz')['ws'][:,:2]
  results={};scales=np.array([1.,.99,.97,.95,.9,.8,.7])
  for collisions in [False,True]:
    task=make_task('two',same_joint_angles=True,finger_collisions=collisions)
    task.model.opt.tolerance=task.model.opt.ls_tolerance=1e-12
    task.model.opt.iterations=task.model.opt.ls_iterations=100
    for name,kappa in [('smooth',1e-5),('zero',0.)]:
      m=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=kappa)
      d=mjx.make_data(m)
      @jax.jit
      def rollout(scale):
        def advance(x,u):
          nd=mjx.step(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:],ctrl=u))
          xn=jp.r_[nd.qpos,nd.qvel,nd.act];return xn,xn
        us=jp.concatenate([jp.asarray(source)*scale,jp.zeros_like(source)],axis=1)
        _,tail=jax.lax.scan(advance,jp.asarray(task.x_0),us)
        return jp.concatenate([task.x_0[None],tail])
      runs=np.asarray(jax.jit(jax.vmap(rollout))(scales))
      rows=[]
      for scale,xs in zip(scales,runs):
        violation=float(np.maximum(np.maximum(task.x_bounds[0]-xs,xs-task.x_bounds[1]),0).max())
        controls=np.hstack([source*scale,np.zeros_like(source)])
        control_violation=float(np.maximum(np.maximum(task.u_bounds[0]-controls,controls-task.u_bounds[1]),0).max())
        rows.append({'control_scale':scale,'angle_degrees':float(np.rad2deg(xs[-1,m.nq-1])),
                     'state_bound_violation':violation,'control_bound_violation':control_violation})
      label=('collisions_' if collisions else 'knob_only_')+name
      results[label]=rows
      np.savez_compressed(ROOT/(label+'_embedded.npz'),scales=scales,xs=runs)
  dump(ROOT/'embedded_one_finger_controls.json',results)
  print(json.dumps(results,default=lambda a:np.asarray(a).tolist(),indent=2))

if __name__=='__main__':main()
