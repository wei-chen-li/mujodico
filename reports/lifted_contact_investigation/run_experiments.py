"""Headless, reproducible contact-lifting ablations; no library files are changed.

Run from the repository root with JAX_ENABLE_X64=True JAX_PLATFORMS=cpu:
  python reports/lifted_contact_investigation/run_experiments.py --help
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import numpy as np

jax.config.update('jax_compilation_cache_dir', '/tmp/mujodico-contact-jax-cache')

from mujodico import trajopt
from mujodico.examples import methods, tasks
from mujodico.mjx._src import solver_sap_util as su


def jsonable(value):
  if isinstance(value, bytes):
    return value.decode()
  if isinstance(value, (np.ndarray, jax.Array)):
    return np.asarray(value).tolist()
  if isinstance(value, np.generic):
    return value.item()
  if hasattr(value, 'value'):
    return value.value
  raise TypeError(type(value).__name__)


def main():
  ap = argparse.ArgumentParser(__doc__)
  ap.add_argument('--task', default='PushSphere2d')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--smoothing', type=float, default=2e-5,
                  help='Actual contact smoothing, after the lifted example multiplier.')
  ap.add_argument('--relu', choices=['reciprocal', 'algebraic', 'stable-algebraic', 'softplus', 'local'],
                  default='reciprocal')
  ap.add_argument('--shooting', action='store_true')
  ap.add_argument('--consistent-smoothing', action='store_true')
  ap.add_argument('--reduced-force-barrier', action='store_true')
  ap.add_argument('--omit-cone', action='store_true')
  ap.add_argument('--constant-init', action='store_true')
  ap.add_argument('--free-forces', action='store_true')
  ap.add_argument('--accept-every', action='store_true')
  ap.add_argument('--elliptic', action='store_true')
  ap.add_argument('--condim', type=int,
                  help='Override the native contact dimension; omit to preserve it.')
  ap.add_argument('--iterations', type=int, default=500)
  ap.add_argument('--mu-init', type=float, default=0.1)
  ap.add_argument('--bound-push', type=float, default=0.01)
  ap.add_argument('--bound-relax', type=float, default=1e-8)
  ap.add_argument('--inner-tolerance', type=float)
  ap.add_argument('--inner-iterations', type=int)
  ap.add_argument('--safe-zero', action='store_true',
                  help='Use an explicit hard-contact branch to isolate sqrt(0) AD failures.')
  ap.add_argument('--exact-hessian', action='store_true')
  ap.add_argument('--label', required=True)
  ap.add_argument('--solver', choices=['ipopt', 'ippas', 'fatrop'], default='ipopt')
  args = ap.parse_args()
  root = Path(__file__).resolve().parent
  output = root / 'data'
  output.mkdir(exist_ok=True)

  # Change both the value and its hand-coded velocity derivative. Changing only
  # one is not an ablation of the contact model: it corrupts the implicit Jacobian.
  if args.relu != 'reciprocal':
    # Repository alternatives contain Python assertions on epsilon that cannot
    # accept SAP's traced arrays. These equivalent formulas also define the
    # epsilon=0 limit, needed for the independent physical replay.
    def alternative(x, epsilon):
      e = jp.where(epsilon > 0, epsilon, 1.)
      if args.relu in ('algebraic', 'stable-algebraic'):
        r = jp.sqrt(x*x + (e/2)**2)
        y = (jp.where(x < 0, (e/2)**2 / (2*(r-x)), (x+r)/2)
             if args.relu == 'stable-algebraic' else (x+r)/2)
      elif args.relu == 'softplus':
        e = e / (4*jp.log(2.))
        y = e * jax.nn.softplus(x/e)
      else:
        y = jp.where(x <= -e, 0., jp.where(x < e, (x+e)**2/(4*e), x))
      return jp.where(epsilon > 0, y, jp.maximum(x, 0.))

    def alternative_grad(x, epsilon):
      e = jp.where(epsilon > 0, epsilon, 1.)
      if args.relu in ('algebraic', 'stable-algebraic'):
        r = jp.sqrt(x*x + (e/2)**2)
        y = (jp.where(x < 0, (e/2)**2/(2*r*(r-x)), (1+x/r)/2)
             if args.relu == 'stable-algebraic' else (1+x/r)/2)
      elif args.relu == 'softplus':
        y = jax.nn.sigmoid(x/(e/(4*jp.log(2.))))
      else:
        y = jp.where(x <= -e, 0., jp.where(x < e, (x+e)/(2*e), 1.))
      return jp.where(epsilon > 0, y, (x > 0).astype(jp.asarray(x).dtype))

    su.reciprocal_approx_relu = alternative
    su.reciprocal_approx_relu_grad = alternative_grad

  if args.safe_zero:
    if args.smoothing != 0:
      raise ValueError('--safe-zero requires --smoothing 0')

    def hard_impulse(vn, phi0, dt, k, d, contact_smoothing, bounce_speed_tol,
                     next_step=True):
      t = dt if next_step else 0.
      return dt*k*d*jp.maximum(-phi0-t*vn, 0.)*su.local_smooth_relu(1/d-vn, bounce_speed_tol)

    def hard_grad(vn, phi0, dt, k, d, contact_smoothing, bounce_speed_tol,
                  next_step=True):
      t = dt if next_step else 0.
      a, b = -phi0-t*vn, 1/d-vn
      return dt*k*d*(-t*(a > 0)*su.local_smooth_relu(b, bounce_speed_tol)
                     - jp.maximum(a, 0.)*su.local_smooth_relu_grad(b, bounce_speed_tol))

    su.hunt_crossley_normal_impulse = hard_impulse
    su.hunt_crossley_normal_impulse_grad = hard_grad

  task = getattr(tasks, args.task)(args.seed)
  if args.inner_tolerance is not None:
    task.model.opt.tolerance = args.inner_tolerance
    task.model.opt.ls_tolerance = args.inner_tolerance
  if args.inner_iterations is not None:
    task.model.opt.iterations = args.inner_iterations
    task.model.opt.ls_iterations = args.inner_iterations
  if args.condim is not None:
    task.model.geom_condim[:] = args.condim
  if args.elliptic:
    import mujoco
    task.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  object.__setattr__(task, 'contact_smoothing',
                     args.smoothing if args.shooting else args.smoothing * 1000)
  started = time.perf_counter()
  trans = (methods.MultipleShootingTranscription(task) if args.shooting else
           methods.LiftedMultipleShootingTranscription(task))
  reduced_force = None
  if args.consistent_smoothing or args.reduced_force_barrier:
    if not args.shooting:
      raise ValueError('Reduced experiments require --shooting')
    f2, eq2, _, wb2 = trajopt.lifted_shooting_transcription(
        task.model, task.contact_approx, args.smoothing, task.u_bounds)
    ncf = len(wb2[0]) - task.nu

    def reduced_force(x, u):
      return -eq2(x, jp.concatenate([u, jp.zeros(ncf)]))

    trans.dynamics = lambda x, u: f2(x, jp.concatenate([u, reduced_force(x, u)]))
  nw = task.nu if args.shooting else trans.nw
  bounds = trans.u_bounds if args.shooting else trans.w_bounds
  if args.free_forces:
    lb = bounds[0].copy()
    lb[task.nu:] = -np.inf
    bounds = lb, bounds[1]

  def equality(x, w, xn, k, p):
    defect = xn - trans.dynamics(x, w)
    return defect if args.shooting else jp.concatenate([defect, trans.stage_equality(x, w)])

  history = []

  def callback(stats, params):
    history.append(stats.copy())
    if stats['iter'] % 25 == 0:
      print(json.dumps({k: stats[k] for k in
                        ['iter', 'objective', 'inf_pr', 'inf_du', 'mu', 'alpha_pr', 'ls_trials']},
                       default=jsonable), flush=True)
    return params

  local_inequality = None
  if args.reduced_force_barrier:
    local_inequality = lambda x, w, xn, k, p: reduced_force(x, w)
  elif not args.shooting and not args.omit_cone:
    local_inequality = lambda x, w, xn, k, p: trans.stage_inequality(x, w)

  solver = trajopt.IpoptTrajOpt(
      N=trans.N, nx=trans.nx, nu=nw,
      local_cost=lambda x, w, xn, k, p: trans.stage_cost(x, w, p),
      terminal_cost=trans.terminal_cost,
      initial_equality_constraint=lambda x, p: x - p.x_0,
      local_equality_constraint=equality,
      local_inequality_constraint=local_inequality,
      x_bounds=trans.x_bounds, u_bounds=bounds, params_trace=trans.params,
      gauss_newton=not args.exact_hessian,
      update_params=callback,
  )
  nlp = solver._nlp
  xs_init = (np.tile(task.x_0, (trans.N + 1, 1)) if args.constant_init else task.xs_init)
  ws_init = np.zeros((trans.N, nw))
  ws_init[:, :task.nu] = task.us_init
  init_flat = nlp._pack(xs_init, ws_init)
  init_constraints = np.asarray(nlp.constraints(init_flat))
  init_objective = float(nlp.objective(init_flat))
  compile_time = time.perf_counter() - started
  if args.solver == 'ipopt':
    xs, ws, info = solver.solve(
        trans.params, xs_init, ws_init, max_iter=args.iterations,
        print_level=0, mu_init=args.mu_init, bound_push=args.bound_push,
        bound_relax_factor=args.bound_relax,
        accept_every_trial_step='yes' if args.accept_every else 'no',
    )
  else:
    ocp_cls = trajopt.IppasTrajOpt if args.solver == 'ippas' else trajopt.FatropTrajOpt
    ocp = ocp_cls(
        N=trans.N, nx=trans.nx, nu=nw,
        initial_state=lambda p: p.x_0,
        dynamics=lambda x, w, k, p: trans.dynamics(x, w),
        stage_cost=lambda x, w, k, p: trans.stage_cost(x, w, p),
        terminal_cost=trans.terminal_cost,
        stage_equality_constraint=None if args.shooting else
            lambda x, w, k, p: trans.stage_equality(x, w),
        stage_inequality_constraint=None if args.shooting else
            lambda x, w, k, p: trans.stage_inequality(x, w),
        x_bounds=trans.x_bounds, u_bounds=bounds, params_trace=trans.params,
    )
    if args.solver == 'ippas':
      started = time.perf_counter()
      xs, ws, info = jax.jit(lambda x, w: ocp.solve(
          trans.params, x, w, max_iter=args.iterations))(
              jp.asarray(xs_init), jp.asarray(ws_init))
      jax.block_until_ready((xs, ws, info))
      info['total_time'] = time.perf_counter() - started
    else:
      xs, ws, info = ocp.solve(trans.params, xs_init, ws_init,
                              max_iter=args.iterations, print_level=0)
    info.pop('warm_start', None)
    xs, ws = np.asarray(xs), np.asarray(ws)
  packed = nlp._pack(xs, ws)
  constraints = np.asarray(nlp.constraints(packed))
  c_lb, c_ub = nlp.constraint_bounds
  violations = np.maximum(np.maximum(c_lb - constraints, constraints - c_ub), 0)
  var_lb, var_ub = nlp.var_bounds
  bound_violation = np.maximum(np.maximum(var_lb - packed, packed - var_ub), 0)

  # Replay the controls using zero smoothing, independently of the lifted states.
  physical_dynamics = trajopt.shooting_transcription(task.model, task.contact_approx, 0.0)

  @jax.jit
  def replay(us):
    def step(x, u):
      xn = physical_dynamics(x, u)
      return xn, xn
    _, tail = jax.lax.scan(step, jp.asarray(task.x_0), us)
    return jp.concatenate([jp.asarray(task.x_0)[None], tail])

  replay_xs = np.asarray(replay(jp.asarray(ws[:, :task.nu])))
  goal_cost = lambda x: float((x - task.x_goal) @ task.Q @ (x - task.x_goal))
  is_knob = args.task in ('PushSquareKnob', 'PushCircularKnob')
  position_error = lambda x: None if is_knob else float(np.linalg.norm(x[2:4] - task.x_goal[2:4]))
  delta = xs[:, :2] if is_knob else xs[:, :2] - xs[:, 2:4]
  initial_delta = task.x_0[:2] if is_knob else task.x_0[:2] - task.x_0[2:4]
  angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
  summary = {
      'args': vars(args), 'jax_version': jax.__version__, 'devices': str(jax.devices()),
      'horizon': trans.N, 'nx': trans.nx, 'nq': task.model.nq,
      'timestep': float(task.model.opt.timestep),
      'geom_condim': np.asarray(task.model.geom_condim), 'cone': str(task.model.opt.cone),
      'x0': task.x_0, 'ncf': nw - task.nu, 'compile_seconds': compile_time,
      'initial_objective': init_objective,
      'initial_constraint_max': float(np.max(np.abs(init_constraints))),
      'info': info, 'iterations': history[-1]['iter'] if history else info.get('iter_count', info.get('iterations')),
      'final_objective': float(nlp.objective(packed)),
      'goal_cost': goal_cost(xs[-1]), 'object_position_error': position_error(xs[-1]),
      'hinge_angle_error_radians': float(abs(xs[-1,2]-task.x_goal[2])) if is_knob else None,
      'replay_hinge_angle_error_radians': float(abs(replay_xs[-1,2]-task.x_goal[2])) if is_knob else None,
      'replay_goal_cost': goal_cost(replay_xs[-1]),
      'replay_object_position_error': position_error(replay_xs[-1]),
      'replay_max_state_error': float(np.max(np.abs(replay_xs - xs))),
      'constraint_violation_max': float(np.max(violations)),
      'dynamics_defect_max': float(np.max(np.abs(
          constraints[trans.nx:trans.nx + trans.N * sum(nlp.constraint_sizes[1:3])]
          .reshape(trans.N, -1)[:, :trans.nx]))),
      'bound_violation_max': float(np.max(bound_violation)),
      'pusher_displacement_max': float(np.max(np.linalg.norm(xs[:, :2] - task.x_0[:2], axis=1))),
      'relative_angle_range_degrees': float(np.ptp(angles) * 180 / np.pi),
      'relative_initial_axis_min': float(np.min(delta @ initial_delta / np.linalg.norm(initial_delta))),
      'backtracked_iterations': sum(h['ls_trials'] > 1 for h in history),
      'fractional_primal_steps': sum(0 < h['alpha_pr'] < .999 for h in history[1:]),
      'restoration_iterations': sum(h['alg_mod'] != 0 for h in history),
  }
  if not args.shooting:
    local = constraints[trans.nx:trans.nx + trans.N * sum(nlp.constraint_sizes[1:3])].reshape(trans.N, -1)
    summary['force_equality_defect_max'] = float(np.max(np.abs(local[:, trans.nx:trans.nx + nw - task.nu])))
    summary['contact_force_min'] = float(np.min(ws[:, task.nu:]))
    summary['contact_force_max'] = float(np.max(ws[:, task.nu:]))
    cf0 = -np.asarray(jax.jit(trans.stage_equality)(jp.asarray(task.x_0), jp.zeros(nw)))
    summary['initial_forward_contact_force'] = cf0
  keep = [i for i, h in enumerate(history)
          if h['iter'] <= 10 or h['iter'] in (20, 50, 100, 200, 300, 400, 500)
          or i == len(history)-1]
  np.savez_compressed(output / (args.label + '.npz'), xs=xs, ws=ws,
                      replay_xs=replay_xs,
                      iterates=np.asarray([history[i]['x'] for i in keep]),
                      iterate_numbers=np.asarray([history[i]['iter'] for i in keep]))
  (output / (args.label + '.json')).write_text(json.dumps(summary, default=jsonable, indent=2))
  stats = [{k: v for k, v in h.items() if k != 'x'} for h in history]
  (output / (args.label + '_history.json')).write_text(json.dumps(stats, default=jsonable, indent=2))
  console_summary = dict(summary)
  console_summary['info'] = {k: v for k, v in info.items()
                             if not isinstance(v, np.ndarray) or v.ndim == 0}
  print('RESULT ' + json.dumps(console_summary, default=jsonable), flush=True)


if __name__ == '__main__':
  main()
