"""IP-PAS (jitted FATROP interior point + parallel associative scan) method."""

import os
import time

import jax

jax.config.update('jax_enable_x64', True)

# Persistent XLA compilation cache: later processes skip most of the compile
# (AllegroBall cold start 63 s -> 19 s).  Respects an existing JAX cache
# setting; set IPPAS_JAX_CACHE='' to disable or to a path to relocate.
_CACHE = os.environ.get('IPPAS_JAX_CACHE',
                        os.path.expanduser('~/.cache/mujodico_ippas'))
if _CACHE and not jax.config.jax_compilation_cache_dir:
  jax.config.update('jax_compilation_cache_dir', _CACHE)
  jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

from mujodico.examples import methods  # noqa: E402
from mujodico.examples.tasks import Task  # noqa: E402
from mujodico.trajopt._src.fatrop_ref import ippas  # noqa: E402


class IppasLiftedMultipleShooting:
  """Lifted multiple shooting solved by IP-PAS.

  The whole interior-point solve runs on the device as one compiled program.
  It is compiled once per task class; ``update(task)`` swaps the runtime
  parameters (initial state, goal, weights) without recompiling.

  Args:
    task: the task.
    gauss_newton: use the objective Hessian only (``'cost'``), as in the
      paper; otherwise the exact Lagrangian Hessian (expensive for MJX).
    smoothing_scale: contact smoothing relative to ``task.contact_smoothing``.
      The other lifted-multiple-shooting methods use ``0.001``.
    linear_backend: ``'pas'`` (parallel associative scan) or ``'serial'``.
    max_iter: compiled iteration cap (``solve(iterations=...)`` may be lower).
    state_bounds: impose the joint-range state bounds of ``task.x_bounds``.
      Off by default: before the ``jnt_type`` fix in ``Task.x_bounds`` these
      were silently empty, which is the setting the paper's Table II was
      produced with, and enabling them lowers success (PushBox2d 100% -> 85%).
    options: overrides for :class:`ippas.IpPasOptions` fields.
  """

  def __init__(
      self,
      task: Task,
      *,
      gauss_newton: bool = True,
      smoothing_scale: float = 0.01,
      linear_backend: str = 'pas',
      max_iter: int = 500,
      state_bounds: bool = False,
      condense_contact: bool = True,
      **options,
  ):
    self.state_bounds = state_bounds
    self.condense_contact = condense_contact
    self._nu = task.nu
    self.transcription = methods.LiftedMultipleShootingTranscription(
        task, contact_smoothing=task.contact_smoothing * smoothing_scale)
    self.hessian_mode = 'cost' if gauss_newton else 'exact_lagrangian'
    # Tuned defaults for the contact benchmarks (see benchmark_ippas.py):
    # FATROP's algorithm unchanged except watchdog off (as in the paper's
    # runs) and a barrier-parameter floor of 1e-2, which was the best or tied
    # best setting on every Table II task.  Pass mu_min=0.0 and
    # watchdog_shortened_iter_trigger=10 for stock FATROP behavior.
    opts = dict(max_iter=max_iter, record_trace=False,
                watchdog_shortened_iter_trigger=0, mu_min=1e-2)
    opts.update(options)
    self.options = ippas.IpPasOptions(**opts)
    self.linear_backend = linear_backend
    self.solver = self._make_solver()
    self._task = task

  def _make_solver(self) -> ippas.IpPasSolver:
    trans = self.transcription
    x_lb, x_ub = trans.x_bounds
    if not self.state_bounds:
      x_lb = np.full_like(x_lb, -np.inf)
      x_ub = np.full_like(x_ub, np.inf)
    w_lb, w_ub = trans.w_bounds
    x_lb_idx = np.flatnonzero(np.isfinite(x_lb))
    x_ub_idx = np.flatnonzero(np.isfinite(x_ub))
    w_lb_idx = np.flatnonzero(np.isfinite(w_lb))
    w_ub_idx = np.flatnonzero(np.isfinite(w_ub))
    n_soc = trans.stage_inequality(jp.zeros(trans.nx), jp.zeros(trans.nw)).size

    # FATROP has no simple bounds: every bound is a slack-backed path
    # inequality ``g(x, w) >= 0``.
    def stage_inequality(x, w):
      values = [
          x[x_lb_idx] - x_lb[x_lb_idx],
          x_ub[x_ub_idx] - x[x_ub_idx],
          w[w_lb_idx] - w_lb[w_lb_idx],
          w_ub[w_ub_idx] - w[w_ub_idx],
          trans.stage_inequality(x, w),
      ]
      return jp.concatenate(values)

    n_ineq = (x_lb_idx.size + x_ub_idx.size + w_lb_idx.size + w_ub_idx.size +
              n_soc)
    return ippas.IpPasSolver(
        horizon=trans.N,
        nx=trans.nx,
        nw=trans.nw,
        stage_cost=lambda x, w, p: trans.stage_cost(x, w, p['p']),
        terminal_cost=lambda x, p: trans.terminal_cost(x, p['p']),
        dynamics=trans.dynamics,
        stage_equality=trans.stage_equality,
        stage_inequality=stage_inequality,
        ineq_lb=np.zeros(n_ineq),
        ineq_ub=np.full(n_ineq, np.inf),
        hessian_mode=self.hessian_mode,
        linear_backend=self.linear_backend,
        options=self.options,
        # cf - f_contact(x, u) = 0 has an identity block in the cf columns:
        # condense the lifted forces out of the KKT solves (exact).
        eq_identity_cols=(np.arange(self._nu, trans.nw)
                          if self.condense_contact else None),
        check_state=np.asarray(trans.params.x_0),
    )

  def update(self, task: Task):
    self.transcription.update(task)
    self._task = task

  def _params(self):
    p = self.transcription.params
    return {'x0': jp.asarray(p.x_0), 'p': p}

  def solve(self, traj_init: dict, *, iterations: int = 500,
            verbose: bool = True):
    trans = self.transcription
    xs_init = np.asarray(traj_init['xs'], dtype=np.float64)
    us_init = np.asarray(traj_init['us'], dtype=np.float64)
    nu = us_init.shape[-1]
    cfs_init = traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu)))
    ws_init = np.concatenate([us_init, cfs_init], axis=1)
    iterations = min(int(iterations), self.options.max_iter)

    tic = time.perf_counter()
    S = self.solver.solve(self._params(), xs_init, ws_init,
                          max_iter=iterations)
    x = np.asarray(S.cur.x)
    w = np.asarray(S.cur.w)
    total_time = time.perf_counter() - tic
    status = ippas.STATUS_NAMES.get(int(S.status), str(int(S.status)))
    if verbose:
      print(f'''
Method: {self.__class__.__name__}
  total time: {total_time:.6f} sec
  solver status: {status}
  iterations: {int(S.it)} (restorations: {int(S.n_resto)})
''')
    return {
        'xs': x,
        'us': w[:, :nu],
        'cfs': w[:, nu:],
        'status': status,
        'iterations': int(S.it),
        'restorations': int(S.n_resto),
    }
