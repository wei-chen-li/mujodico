"""Static configuration for the compiled IPPAS solver."""

from dataclasses import dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class Options:
  """FATROP defaults, with bounded device-side search loops.

  Options are static under JIT: close over them or mark them static_argnames.
  Bounds use FATROP's configurable relaxation. Watchdog and restoration are
  not implemented.
  """

  max_iter: int = 1000
  tol: float = 1e-8
  tol_acceptable: float = 1e-6
  acceptable_iter: int = 15
  constr_viol_tol: float = 1e-4
  mu_init: float = 0.1
  barrier_tol_factor: float = 10.0
  mu_linear_decrease_factor: float = 0.2
  mu_superlinear_decrease_power: float = 1.5
  compl_inf_tol: float = 1e-4
  tau_min: float = 0.99
  bound_push: float = 1e-2
  bound_frac: float = 1e-2
  bound_relax_factor: float = 1e-8
  initialize_duals: bool = True
  lam_max: float = 1e3
  kappa_d: float = 1e-5
  kappa_sigma: float = 1e10
  s_max: float = 100.0
  max_ls: int = 40
  max_soc: int = 4
  kappa_soc: float = 0.99
  alpha_red_factor: float = 0.5
  alpha_min_frac: float = 0.05
  theta_max_fact: float = 1e4
  theta_min_fact: float = 1e-4
  s_phi: float = 2.3
  s_theta: float = 1.1
  delta: float = 1.0
  gamma_theta: float = 1e-5
  gamma_phi: float = 1e-8
  eta_phi: float = 1e-8
  obj_max_incr: float = 5.0
  filter_reset_trigger: int = 5
  max_filter_resets: int = -1
  delta_w0: float = 1e-4
  delta_wmin: float = 1e-20
  delta_w_max: float = 1e40
  kappa_wmin: float = 1.0 / 3.0
  kappa_wplus: float = 8.0
  kappa_wplusem: float = 100.0
  max_regularization: int = 24
  max_refinement: int = 3
  linear_tol: float = 1e-10
  rank_tol: float = 1e-12
  print_level: int = 0

  def __post_init__(self):
    if (
        isinstance(self.max_filter_resets, bool)
        or not isinstance(self.max_filter_resets, Integral)
        or self.max_filter_resets < -1
    ):
      raise ValueError('max_filter_resets must be an integer >= -1.')
    if not isinstance(self.initialize_duals, bool):
      raise ValueError('initialize_duals must be a bool.')
    counts = (
        'max_iter',
        'acceptable_iter',
        'max_soc',
        'print_level',
        'max_refinement',
    )
    for name in (
        *counts,
        'max_ls',
        'max_regularization',
        'filter_reset_trigger',
    ):
      value = getattr(self, name)
      minimum = 0 if name in counts else 1
      if (
          isinstance(value, bool)
          or not isinstance(value, Integral)
          or value < minimum
      ):
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    for name, value in vars(self).items():
      if name in ('initialize_duals', 'max_filter_resets'):
        continue
      if not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
      if name in ('kappa_d', 'bound_relax_factor') and value == 0:
        continue
      if name not in counts and value <= 0:
        raise ValueError(f'{name} must be positive.')
    for name in (
        'tau_min',
        'bound_frac',
        'alpha_red_factor',
        'alpha_min_frac',
        'gamma_theta',
        'gamma_phi',
        'eta_phi',
        'kappa_soc',
        'mu_linear_decrease_factor',
        'kappa_wmin',
    ):
      if not 0 < getattr(self, name) < 1:
        raise ValueError(f'{name} must lie strictly between zero and one.')
    if self.bound_frac >= 0.5:
      raise ValueError('bound_frac must be less than 0.5.')
    if self.mu_superlinear_decrease_power <= 1:
      raise ValueError('mu_superlinear_decrease_power must exceed one.')
    if self.delta_wmin > self.delta_w0 or self.delta_w0 > self.delta_w_max:
      raise ValueError('Require delta_wmin <= delta_w0 <= delta_w_max.')
    if min(self.kappa_wplus, self.kappa_wplusem) <= 1:
      raise ValueError('Regularization growth factors must exceed one.')
