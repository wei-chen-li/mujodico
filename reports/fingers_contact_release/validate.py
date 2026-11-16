"""Independent original-model replay, bounds, cost, and collision checks."""
import argparse
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import numpy as np
from mujodico import mjx
from mujodico.examples.tasks import FingersSphere2d
from mujodico.trajopt._src.transcriptions import _put_optimization_model
from run import DATA,ROOT,save_json

jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')


def main():
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--label',default='ms_physical_limit');args=ap.parse_args()
  task=FingersSphere2d();raw=np.load(DATA/(args.label+'.npz'));us=raw['ws'][:,:task.nu]
  results={}
  for name,tol,n in [('default',1e-5,20),('tight',1e-12,100)]:
    task.model.opt.tolerance=task.model.opt.ls_tolerance=tol
    task.model.opt.iterations=task.model.opt.ls_iterations=n
    m=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=0.)
    d=mjx.make_data(m)
    @jax.jit
    def rollout(us):
      def advance(x,u):
        nd=mjx.step(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:],ctrl=u))
        xn=jp.concatenate([nd.qpos,nd.qvel,nd.act]);return xn,xn
      _,tail=jax.lax.scan(advance,jp.asarray(task.x_0),us)
      return jp.concatenate([jp.asarray(task.x_0)[None],tail])
    # Enforce the literal original control bounds before the independent replay.
    controls=np.clip(us,*task.u_bounds);xs=np.asarray(rollout(jp.asarray(controls)))
    e=xs-task.x_goal
    J=float(np.einsum('ki,ij,kj->',e[:-1],task.Q,e[:-1])+e[-1]@task.Qf@e[-1]
       +np.einsum('ki,ij,kj->',controls,task.R,controls))
    pairs=np.asarray(mjx.fwd_position(m,d)._impl.contact.geom)
    assert pairs.shape==(4,2) and np.all(np.any(pairs==4,axis=1)),pairs
    assert np.isfinite(xs).all()
    results[name]={'objective':J,'angle_degrees':float(np.degrees(xs[-1,6])),
      'position_error_mm':float(np.linalg.norm(xs[-1,4:6]-task.x_goal[4:6])*1000),
      'state_bound_violation':float(np.maximum(np.maximum(task.x_bounds[0]-xs,xs-task.x_bounds[1]),0).max()),
      'control_bound_violation_before_clip':float(np.maximum(np.maximum(task.u_bounds[0]-us,us-task.u_bounds[1]),0).max()),
      'state_difference_from_saved_zero_replay':float(abs(xs-raw['zero_xs']).max()),
      'contact_pairs':pairs,'inner_iterations':n,'inner_tolerance':tol}
  results['label']=args.label
  results['initial_state_control_hash']=hashlib.sha256(task.xs_init.tobytes()+task.us_init.tobytes()).hexdigest()
  save_json(ROOT/'validation.json',results)
  for k,v in results.items():print(k,v)


if __name__=='__main__':main()
