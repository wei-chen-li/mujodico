"""Contact-tail, representation, and finite-difference diagnostics."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from mujodico import mjx, trajopt
from mujodico.examples.tasks import PushSphere2d, PushBox2d
from mujodico.mjx._src import solver_sap_util as su
from mujodico.trajopt._src.transcriptions import _extract_contact_indices

jax.config.update('jax_compilation_cache_dir', '/tmp/mujodico-contact-jax-cache')
output = Path(__file__).resolve().parent / 'data'
output.mkdir(exist_ok=True)
results = {}

for task_cls in [PushSphere2d, PushBox2d]:
  task = task_cls(0)
  per_task = []
  for smoothing in [2e-5, 2e-8, 2e-11, 0.0]:
    dyn, eq, ineq, bounds = trajopt.lifted_shooting_transcription(
        task.model, task.contact_approx, smoothing, task.u_bounds)
    x = jp.asarray(task.x_0)
    w = jp.zeros(len(bounds[0]))
    force = lambda x: -eq(x, w)
    cf = np.asarray(jax.jit(force)(x))
    df = np.asarray(jax.jit(jax.jacfwd(force))(x))
    dx_zero, dw_zero = jax.jit(jax.jacfwd(dyn, (0, 1)))(x, w)
    dx_pushed, dw_pushed = jax.jit(jax.jacfwd(dyn, (0, 1)))(x, w.at[task.nu:].set(.01))
    delta = 1e-6
    fd = (np.asarray(force(x.at[0].add(delta))) -
          np.asarray(force(x.at[0].add(-delta)))) / (2 * delta)
    row = {'smoothing': smoothing, 'force': cf.tolist(),
           'force_dx': df.tolist(), 'force_fd_pusher_x': fd.tolist(),
           'log_force_gradient_pusher': (df[:, :2] / cf[:, None]).tolist(),
           'relaxed_log_force_gradient_pusher': (df[:, :2] / (cf[:, None] + 1e-8)).tolist(),
           'dynamics_object_velocity_dforce': np.asarray(dw_zero)[task.model.nq+2:task.model.nq+4, task.nu:].tolist(),
           'dynamics_object_velocity_dpusher_zero_force': np.asarray(dx_zero)[task.model.nq+2:task.model.nq+4, :2].tolist(),
           'dynamics_object_velocity_dpusher_pushed_force': np.asarray(dx_pushed)[task.model.nq+2:task.model.nq+4, :2].tolist()}
    per_task.append(row)
    print(task_cls.__name__, smoothing, cf, 'dlogF/dq', row['log_force_gradient_pusher'], flush=True)
  results[task_cls.__name__] = per_task
  (output / 'derivative_probes.json').write_text(json.dumps(results, indent=2))

# condim=1 is routed through CONTACT_FRICTIONLESS for both cone options.
representations = []
for condim in [1, 3]:
  for cone in [mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC]:
    task = PushSphere2d()
    task.model.geom_condim[:] = condim
    task.model.opt.cone = cone
    model = mjx.put_model(task.model, contact_approximation=task.contact_approx,
                          contact_smoothing=2e-5)
    try:
      data = mjx.make_data(model)
    except NotImplementedError as error:
      representations.append({'condim': condim, 'cone': str(cone), 'error': str(error)})
      continue
    data = jax.jit(mjx.forward)(model, data.replace(qpos=jp.asarray(task.x_0[:model.nq])))
    addresses, contacts = _extract_contact_indices(model)
    _, eq, ineq, bounds = trajopt.lifted_shooting_transcription(
        task.model, task.contact_approx, 2e-5, task.u_bounds)
    force = np.asarray(data._impl.efc_force)[addresses]
    row = {'condim': condim, 'cone': str(cone), 'ncf': len(addresses),
           'types': [str(c.type) for c in contacts], 'force': force.tolist(),
           'cone_residual_at_force': np.asarray(ineq(jp.asarray(task.x_0), jp.concatenate([jp.zeros(task.nu), jp.asarray(force)]))).tolist(),
           'cone_residual_at_zero': np.asarray(ineq(jp.asarray(task.x_0), jp.zeros(len(bounds[0])))).tolist(),
           'contact_jacobian': np.asarray(data._impl.efc_J)[addresses].tolist()}
    representations.append(row)
    print('representation', row, flush=True)
results['representations'] = representations

tails = []
for epsilon in [1e-3, 1e-5, 1e-8, 1e-10]:
  x = jp.asarray(-.4)
  for name in ['reciprocal_approx_relu', 'algebraic_smooth_relu', 'softplus', 'local_smooth_relu']:
    fn = getattr(su, name)
    value = float(fn(x, epsilon))
    derivative = float(getattr(su, name + '_grad')(x, epsilon))
    autodiff = float(jax.grad(lambda x: fn(x, epsilon))(x))
    tails.append({'epsilon': epsilon, 'name': name, 'value': value,
                  'derivative': derivative, 'autodiff': autodiff,
                  'dlog': derivative / value if value else None})
results['relu_tails'] = tails
(output / 'derivative_probes.json').write_text(json.dumps(results, indent=2))

params = dict(vn=jp.array(-.1), phi0=jp.array(0.), dt=jp.array(.02),
              k=jp.array(1e5), d=jp.array(5.), bounce_speed_tol=.001)
scalars = {'touching_approaching_impulses': {
    str(k): float(su.hunt_crossley_normal_impulse(contact_smoothing=k, **params))
    for k in [0., 2e-5, 2e-11, 2e-17]}}
separated = lambda k: su.hunt_crossley_normal_impulse(
    vn=jp.array(0.), phi0=jp.array(.4), dt=.02, k=k, d=5.,
    contact_smoothing=0., bounce_speed_tol=.001)
scalars['zero_smoothing_separated_derivative_k'] = str(float(jax.grad(separated)(jp.array(1e5))))
scalars['zero_smoothing_separated_fd_k'] = float((separated(1e5+1)-separated(1e5-1))/2)
(output / 'scalar_probes.json').write_text(json.dumps(scalars, indent=2))
