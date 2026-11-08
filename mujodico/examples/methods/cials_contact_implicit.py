"""Example adapter for the two multiplier-update schedules of CIALS."""

from dataclasses import replace
import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico import trajopt
from mujodico.examples.methods.transcription_contact_implicit import ContactImplicitTranscription
from mujodico.examples.tasks import Task


class CialsContactImplicit:
  """Contact-implicit transcription solved by CIALS.

  Pass ``options=CIALSTrajOpt.Options(variant=CIALSTrajOpt.Variant.ADMM)``
  to select one primal sweep per multiplier update. ``iterations`` in solve
  limits multiplier updates, while options.ipopt_max_iter limits each NLP.
  """

  def __init__(
      self, task: Task, *, options: trajopt.CIALSTrajOpt.Options | None = None
  ):
    self.transcription = ContactImplicitTranscription(
        task, independent_next_state=True
    )
    self.options = options or trajopt.CIALSTrajOpt.Options()
    trans = self.transcription
    self.solver = trajopt.CIALSTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nw,
        local_cost=lambda x, w, xn, k, p: trans.stage_cost(x, w, p),
        terminal_cost=trans.terminal_cost,
        initial_equality_constraint=lambda x, p: x - p.x_0,
        local_equality_constraint=lambda x, w, xn, k, p: xn
        - trans.dynamics(x, w),
        local_inequality_constraint=lambda x, w, xn, k, p: trans.local_inequality(
            x, w, xn
        ),
        local_complementarity_constraint=trans.local_complementarity,
        x_bounds=trans.x_bounds,
        u_bounds=trans.w_bounds,
        params_trace=trans.params,
    )
    self._solve = jax.jit(self.solver.solve, static_argnames=('options',))

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(
      self, traj_init: dict, *, iterations: int = 100, verbose: bool = True
  ):
    trans = self.transcription
    nu = trans._task.nu
    us = jp.asarray(traj_init['us'])
    cfs = jp.asarray(traj_init.get('cfs', np.zeros((trans.N, trans.nw - nu))))
    ws = jp.concatenate((us, cfs), axis=1)
    options = replace(self.options, max_iterations=iterations)
    start = time.perf_counter()
    xs, ws, info = jax.block_until_ready(
        self._solve(
            trans.params,
            jp.asarray(traj_init['xs']),
            ws,
            options=options,
        )
    )
    elapsed = time.perf_counter() - start
    if verbose:
      status = trajopt.CIALSTrajOpt.Status(int(info['status'])).name
      print(
          f'Method: {self.__class__.__name__} ({options.variant.value})\n'
          f'  status: {status}\n'
          f'  multiplier updates: {int(info["iterations"])}\n'
          f'  trajectory solves: {int(info["sweeps"])}\n'
          f'  total time: {elapsed:.6f} sec'
      )
    return {
        'xs': np.asarray(xs),
        'us': np.asarray(ws[:, :nu]),
        'cfs': np.asarray(ws[:, nu:]),
        'info': info,
    }
