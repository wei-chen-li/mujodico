import jax
import jax.numpy as jp

from mujodico.trajopt._src.ippas import ip_iterate
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import all_finite
from mujodico.trajopt._src.ippas.options import Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel


def check(
    model: StagewiseOcpModel,
    it: ip_iterate.Iterate,
    system: pd.IpPrimalDualSystem,
    data: ip_iterate.IterateData,
    metrics: ip_iterate.Metrics,
    count: jax.Array,
    iteration: jax.Array,
    bounds: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    options: Options,
) -> jax.Array:
  """Return a scalar int32 status for convergence, limits, or numerical failure."""
  from mujodico.trajopt._src.ippas.ip_algorithm import Status

  o = options
  status = jp.where(iteration >= o.max_iter, Status.MAX_ITER, Status.RUNNING)
  feasible = metrics.primal <= o.constr_viol_tol
  status = jp.where(
      (o.acceptable_iter > 0) & (count >= o.acceptable_iter) & feasible,
      Status.ACCEPTABLE,
      status,
  )
  status = jp.where((metrics.error <= o.tol) & feasible, Status.SUCCESS, status)
  status = jp.where(
      ip_iterate.valid(it, bounds, model.quaternion_positions)
      & all_finite((
          system.A,
          system.B,
          system.Ex,
          system.Eu,
          system.Gx,
          system.Gu,
          system.Gf,
          system.E0,
          data,
          metrics,
      )),
      status,
      Status.NUMERICAL_ERROR,
  )
  return status.astype(jp.int32)
