"""Compiled IPPAS multiple shooting with physical controls."""

import time

import jax
import jax.numpy as jp
import numpy as np

from mujodico import trajopt
from mujodico.examples.methods.transcription_multiple_shooting import MultipleShootingTranscription
from mujodico.examples.tasks import Task


class IppasMultipleShooting:

  def __init__(self, task: Task, *, gauss_newton: bool = True):
    self.transcription = MultipleShootingTranscription(task)
    trans = self.transcription
    self.solver = trajopt.IppasTrajOpt(
        N=trans.N,
        nx=trans.nx,
        nu=trans.nu,
        initial_state=lambda params: params.x_0,
        dynamics=lambda x, u, k, params: trans.dynamics(x, u),
        stage_cost=lambda x, u, k, params: trans.stage_cost(x, u, params),
        terminal_cost=trans.terminal_cost,
        x_bounds=trans.x_bounds,
        u_bounds=trans.u_bounds,
        quaternion_positions=task.quaternion_w_idxs,
        gauss_newton=gauss_newton,
        params_trace=trans.params,
    )
    self._solver_cache = {}
    self.info = None

  def update(self, task: Task):
    self.transcription.update(task)

  def solve(
      self,
      traj_init: dict,
      *,
      iterations: int = 500,
      max_soc: int = 4,
      verbose: bool = True,
  ):
    """Return xs/us using a compiled solve cached for each static option set."""
    trans = self.transcription
    xs = jp.asarray(traj_init['xs'], dtype=jp.float64)
    us = jp.asarray(traj_init['us'], dtype=jp.float64)
    key = iterations, max_soc
    if key not in self._solver_cache:
      self._solver_cache[key] = jax.jit(
          lambda p, x, u: self.solver.solve(
              p, x, u, max_iter=iterations, max_soc=max_soc
          )
      )
    started = time.perf_counter()
    xs, us, info = self._solver_cache[key](trans.params, xs, us)
    jax.block_until_ready((xs, us, info))
    self.info = info
    if verbose:
      print(
          f'Method: {self.__class__.__name__}\n'
          f'  total time: {time.perf_counter() - started:.6f} sec\n'
          f'  solver status: {int(info["status"])}\n'
          f'  iterations: {int(info["iter_count"])}\n'
      )
    return {'xs': np.asarray(xs), 'us': np.asarray(us)}
