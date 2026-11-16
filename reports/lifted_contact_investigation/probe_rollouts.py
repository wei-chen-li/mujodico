"""Separate optimization infeasibility, smoothing bias, and cone inconsistency."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujodico import mjx, trajopt
from mujodico.examples import tasks
from mujodico.trajopt._src.transcriptions import _extract_contact_indices

jax.config.update('jax_compilation_cache_dir', '/tmp/mujodico-contact-jax-cache')
root = Path(__file__).resolve().parent / 'data'
results = {}

for label in ['sphere_baseline', 'sphere_no_linesearch', 'sphere_ippas', 'box_baseline', 'box_fatrop',
              'circular_baseline', 'square_baseline', 'circular_elliptic', 'square_elliptic',
              'circular_algebraic', 'square_algebraic', 'circular_no_linesearch', 'circular_ippas',
              'circular_elliptic_no_cone', 'square_elliptic_no_cone']:
  config = root / (label + '.json')
  if not config.exists():
    continue
  summary = json.loads(config.read_text())
  if summary['args']['relu'] != 'reciprocal':
    # Alternative-smoother replay requires its process-local constitutive law;
    # these probes intentionally compare only the repository's native law.
    continue
  task = getattr(tasks, summary['args']['task'])(summary['args']['seed'])
  if summary['args'].get('condim') is not None:
    task.model.geom_condim[:] = summary['args']['condim']
  if summary['args']['elliptic']:
    task.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model = mjx.put_model(task.model, contact_approximation=task.contact_approx,
                        contact_smoothing=summary['args']['smoothing'])
  model = model.tree_replace({'opt.disableflags': model.opt.disableflags | mjx.DisableBit.CLAMPCTRL})
  data = mjx.make_data(model)
  raw = np.load(root / (label + '.npz'))

  @jax.jit
  def replay(us):
    def step(x, u):
      d = data.replace(qpos=x[:model.nq], qvel=x[model.nq:model.nq+model.nv], ctrl=u)
      d = mjx.step(model, d)
      xn = jp.concatenate([d.qpos, d.qvel, d.act])
      return xn, xn
    _, tail = jax.lax.scan(step, jp.asarray(task.x_0), us)
    return jp.concatenate([jp.asarray(task.x_0)[None], tail])

  xs = np.asarray(replay(jp.asarray(raw['ws'][:, :task.nu])))
  optimized = raw['xs']
  is_knob = summary['args']['task'] in ('PushCircularKnob', 'PushSquareKnob')
  results[label] = {
      'smooth_replay_max_state_difference_from_optimized': float(np.max(np.abs(xs-optimized))),
      'smooth_vs_zero_replay_max_state_difference': float(np.max(np.abs(xs-raw['replay_xs']))),
  }
  if is_knob:
    results[label]['smooth_replay_hinge_error_radians'] = float(abs(xs[-1,2]-task.x_goal[2]))
    results[label]['smooth_vs_zero_replay_final_hinge_difference_radians'] = float(abs(xs[-1,2]-raw['replay_xs'][-1,2]))
  else:
    results[label]['smooth_replay_position_error'] = float(np.linalg.norm(xs[-1, 2:4]))
    results[label]['smooth_vs_zero_replay_final_object_position_difference'] = float(np.linalg.norm(xs[-1, 2:4]-raw['replay_xs'][-1, 2:4]))
  np.savez_compressed(root / (label + '_smooth_replay.npz'), xs=xs)
  print(label, results[label], flush=True)
  (root / 'rollout_probes.json').write_text(json.dumps(results, indent=2))

task = tasks.PushSphere2d()
task.model.geom_condim[:] = 3
task.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
model = mjx.put_model(task.model, contact_approximation=task.contact_approx, contact_smoothing=2e-5)
data = mjx.make_data(model)
addresses, _ = _extract_contact_indices(model)

@jax.jit
def forces(x, u):
  d = data.replace(qpos=x[:4], qvel=x[4:], ctrl=u)
  d = mjx.forward(model, d)
  cf = d._impl.efc_force[addresses]
  friction = d._impl.contact.friction[0, :2]
  cone = cf[0]**2 - jp.sum((cf[1:]/friction)**2)
  return cf, cone

states = []
controls = []
for px in [-.12, -.13, -.14, -.3]:
  for vx in [-1., 0., 1.]:
    for ux in [-1., 0., 1.]:
      for uy in [-1., 0., 1.]:
        states.append([px, 0., 0., 0., vx, 0., 0., 0.])
        controls.append([ux, uy])
cf, cone = jax.jit(jax.vmap(forces))(jp.asarray(states), jp.asarray(controls))
cf, cone = np.asarray(cf), np.asarray(cone)
worst = int(np.argmin(cone))
results['elliptic_cone_consistency'] = {
    'number_of_samples': len(states),
    'negative_residual_samples_below_1e-10': int(np.sum(cone < -1e-10)),
    'worst_residual': float(cone[worst]), 'state': states[worst],
    'control': controls[worst], 'forward_force': cf[worst].tolist(),
}
print(results['elliptic_cone_consistency'], flush=True)
(root / 'rollout_probes.json').write_text(json.dumps(results, indent=2))

for task_class, label in [(tasks.PushCircularKnob, 'circular'),
                           (tasks.PushSquareKnob, 'square')]:
  task = task_class()
  task.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model = mjx.put_model(task.model, contact_approximation=task.contact_approx,
                        contact_smoothing=1e-5)
  data = mjx.make_data(model)
  addresses, _ = _extract_contact_indices(model)

  @jax.jit
  def knob_forces(x, u):
    d = mjx.forward(model, data.replace(qpos=x[:3], qvel=x[3:], ctrl=u))
    cf = d._impl.efc_force[addresses]
    friction = d._impl.contact.friction[0, :2]
    return cf, cf[0]**2-jp.sum((cf[1:]/friction)**2)

  states, controls = [], []
  for px in [-.09, -.10, -.11, -.3]:
    for vx in [-1., 0., 1.]:
      for ux in [-.2, 0., .2]:
        for uy in [-.2, 0., .2]:
          states.append([px, 0., 0., vx, 0., 0.])
          controls.append([ux, uy])
  cf, cone = jax.jit(jax.vmap(knob_forces))(jp.asarray(states), jp.asarray(controls))
  cf, cone = np.asarray(cf), np.asarray(cone)
  worst = int(np.argmin(cone))
  results[label + '_elliptic_cone_consistency'] = {
      'number_of_samples': len(states),
      'negative_residual_samples_below_1e-10': int(np.sum(cone < -1e-10)),
      'worst_residual': float(cone[worst]), 'state': states[worst],
      'control': controls[worst], 'forward_force': cf[worst].tolist(),
  }
  print(label, results[label + '_elliptic_cone_consistency'], flush=True)
  (root / 'rollout_probes.json').write_text(json.dumps(results, indent=2))
