"""Check lifting/forward-step equivalence at finite inner-solver accuracy."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import numpy as np
from mujodico import trajopt
from mujodico.examples import tasks

jax.config.update('jax_compilation_cache_dir', '/tmp/mujodico-contact-jax-cache')
root = Path(__file__).resolve().parent / 'data'
results = {}
for label in ['sphere_baseline', 'box_baseline', 'circular_baseline', 'square_baseline']:
  s = json.loads((root/(label+'.json')).read_text())
  task = getattr(tasks, s['args']['task'])(s['args']['seed'])
  f, eq, _, bounds = trajopt.lifted_shooting_transcription(
      task.model, task.contact_approx, s['args']['smoothing'], task.u_bounds)
  ncf = len(bounds[0])-task.nu
  a = np.load(root/(label+'.npz'))

  def composed(x, u):
    cf = -eq(x, jp.concatenate([u, jp.zeros(ncf)]))
    return f(x, jp.concatenate([u, cf]))

  @jax.jit
  def replay(us):
    def step(x, u):
      xn = composed(x, u)
      return xn, xn
    _, xs = jax.lax.scan(step, jp.asarray(task.x_0), us)
    return jp.concatenate([jp.asarray(task.x_0)[None], xs])

  xs = np.asarray(replay(jp.asarray(a['ws'][:,:task.nu])))
  one_step = np.asarray(jax.jit(jax.vmap(composed))(
      jp.asarray(a['xs'][:-1]), jp.asarray(a['ws'][:,:task.nu])))
  native = np.load(root/(label+'_smooth_replay.npz'))['xs']
  row = {
      'composed_replay_max_state_difference_from_optimized': float(np.max(np.abs(xs-a['xs']))),
      'composed_replay_max_state_difference_from_native_replay': float(np.max(np.abs(xs-native))),
      'composed_one_step_defect_max': float(np.max(np.abs(one_step-a['xs'][1:]))),
  }
  if 'Knob' in s['args']['task']:
    row['composed_replay_hinge_error_radians'] = float(abs(xs[-1,2]-task.x_goal[2]))
  else:
    row['composed_replay_position_error_m'] = float(np.linalg.norm(xs[-1,2:4]-task.x_goal[2:4]))
  results[label] = row
  np.savez_compressed(root/(label+'_composed_replay.npz'), xs=xs)
  print(label,row,flush=True)
  (root/'step_equivalence_probes.json').write_text(json.dumps(results,indent=2))
