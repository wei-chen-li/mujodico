# IPPAS

The solver takes a `StagewiseOcpModel` directly and uses the existing
`IpPrimalDualSystem` and parallel equality-constrained LQR implementation.
Neither linear kernel is modified. The nonlinear layer supplies filter
backtracking and filter resets, switching and Armijo tests, second-order
corrections using cached factors, monotone barrier updates, primal
regularization, and iterative
refinement. Watchdog and restoration phases are deferred.

## Components

Each component can be exercised directly without constructing the solver.
`IppasSolver` in `ip_algorithm.py` validates the model and coordinates these
components; it does not implement their numerical policies itself.

| Module | Responsibility | FATROP counterpart |
| --- | --- | --- |
| `ip_algorithm.py` | Public solver and compiled iteration controller | `ip_algorithm.hxx` |
| `ip_nlp_orig.py` | Model evaluations, derivatives, bound relaxation, Newton assembly | `ip_nlp_orig.hxx` |
| `ip_iterate.py` | Primal-dual data, residuals, barrier merit, fraction-to-boundary and trial updates | `ip_iterate.hxx` |
| `ip_initializer.py` | Interior slacks, multiplier initialization, warm starts | `ip_initializer.hxx`, `ip_eq_mult_initializer.hxx` |
| `ip_search_dir.py` | Newton directions, primal curvature correction, full Newton residual checks, and cached-factor refinement | `ip_search_dir.hxx`, `linear_algebra/linear_solver.hxx` |
| `ip_linesearch.py` | Switching/Armijo tests, backtracking, SOC, filter-reset policy | `ip_linesearch.hxx` |
| `ip_filter.py` | Filter storage, strict dominance, raw and margin-shifted insertion | `ip_filter.hpp`, `ip_filter.cpp` |
| `ip_mu_update.py` | Monotone barrier parameter update | `ip_mu_update.hxx` |
| `ip_convergence_check.py` | KKT convergence and termination status | `ip_convergence_check.hxx` |

The package exports its public solver and state types directly from
`ip_algorithm.py` and `ip_iterate.py`. `options.py` remains the static
configuration module; FATROP likewise keeps options outside its `ip_*`
components.

```python
import jax
import jax.numpy as jp
from mujodico.trajopt._src.ippas import IppasSolver, InitialGuess, Options
from mujodico.trajopt._src.stagewise_ocp_model import StagewiseOcpModel

jax.config.update('jax_enable_x64', True)
model = StagewiseOcpModel(
    20, 1, 1,
    initial_state=lambda p: p['x0'],
    dynamics=lambda x, u, k, p: x + u,
    stage_cost=lambda x, u, k, p: 0.5 * jp.sum(u*u),
    terminal_cost=lambda x, p: 5 * jp.sum((x-p['goal'])**2),
)
solver = IppasSolver(model, Options(max_iter=100))
solve = jax.jit(solver.solve)
params = {'x0': jp.array([0.]), 'goal': jp.array([2.])}
result = solve(params, InitialGuess(jp.zeros((21, 1)), jp.zeros((20, 1))))
```

`solver.solve(params, xs_init, us_init)` is equivalent. The result is a JAX
dataclass containing status, objective, iteration count, final residuals,
barrier parameter, and the complete primal-dual iterate. Passing that iterate
as the initial guess performs a warm start. To reuse its duals with different
trajectories, pass `InitialGuess(xs, us, result.iterate)`.

`mujodico.trajopt.IppasTrajOpt` mirrors `FatropTrajOpt` construction, omitting
terminal equalities and parameter-update callbacks. Its
`solve(params, xs_init, us_init, *, warm_start=None, **options)` returns
`(xs, us, info)` with JAX arrays and JAX-valued diagnostics, including an
`info['warm_start']` iterate. The entire call supports `jax.jit` and `jax.vmap`.
Options are static: close over them in a compiled function, or use
`static_argnames` for option keyword arguments. Parameters and guesses remain
dynamic. Enable float64 before initializing the model. Time the solve outside
JIT and synchronize using `jax.block_until_ready`.

Bounds follow FATROP: bounded slacks remain interior; primal trajectories can
be infeasible during iteration. Default relaxation is
`min(constr_viol_tol, 1e-8 * max(1, abs(bound)))` on each finite bound side.
Set `bound_relax_factor=0` to disable relaxation. KKT errors target the relaxed
NLP; reported primal error also includes original bound violations and must
satisfy `constr_viol_tol`. Independent multipliers for `g-s=0` and
slack bounds use primal and dual step lengths respectively. The Hessian
selection is owned entirely by the model's `gauss_newton` setting.

The initial implementation requires `N>=1`, `nu>=1`, no terminal equalities,
and full-row-rank control Jacobians for stage equalities. Unsupported static
structure raises `ValueError`; numerical rank loss returns
`Status.UNSUPPORTED_EQUALITIES`. State-only and redundant equality blocks are
unsupported by this unchanged backend. Quaternion states and both dynamics
Jacobian conventions are supported; finite bounds on quaternion storage
components are currently rejected. Model callbacks must be valid at infeasible
trial points or return nonfinite values that line search can reject.

The primal curvature check is sufficient but conservative: it can regularize
more than FATROP's reduced Riccati inertia checks. Numerical equality with
FATROP's iterations is not promised. No solution-sensitivity API is provided.

Run the local tests with the repository's Python environment:

```bash
JAX_PLATFORMS=cpu python -m unittest discover \
  -s mujodico/trajopt/_src/ippas -p '*_test.py'
```

Component tests are colocated in matching `ip_*_test.py` files. Eleven native
test cases are ported from the supplied FATROP checkout:

| Upstream test file | IPPAS test file | Ported cases |
| --- | --- | --- |
| `unittest/ip_algorithm/ip_filter_test.cpp` | `ip_filter_test.py` | Default constructor, reset, reserve, empty acceptance, augment/acceptance, size |
| `unittest/ip_algorithm/ip_initializer_test.cpp` | `ip_initializer_test.py` | Equality multiplier initialization, lower-bound push, upper-bound push |
| `unittest/ocp/ip_search_dir_test.cpp` | `ip_search_dir_test.py` | Solve linear system, update iterate and check primal/dual infeasibility and linearized complementarity |

The shared point-mass fixture adapts
`unittest/ocp/ocp_test_probem.hpp`: it retains 99 transitions, the time step,
affine dynamics offsets, force costs, initial state, and stage-dependent force
bounds. The latter use normalized inequalities to fit fixed bound metadata.
Terminal equalities are omitted because this backend does not support them.
Initializer push tests supply explicit low/high primal guesses rather than ask
a native NLP callback to overwrite stored values. These adaptations are recorded
in the corresponding test files; the native assertions are preserved.

This checkout has no standalone unit tests for line search, barrier updates,
iterate metrics, or convergence checks. The IPPAS-specific tests cover Armijo
switching, Maratos SOC, filter resets, fraction-to-boundary, distinct primal/dual
step lengths, repeated barrier reductions with fresh residuals, termination
boundaries, negative-curvature correction, and factor reuse. Restoration tests
are deferred with the restoration implementation.

The example method `IppasLiftedMultipleShooting` uses this implementation and
the same lifted transcription and smoothing as `FatropLiftedMultipleShooting`.
It caches compiled solves and preserves the `xs/us/cfs` example interface.

See `NOTICE.md` and `licenses/` for upstream policy attribution.

CPU validation on `PushSphere2d(seed=0)` with the unchanged task and a
500-iteration budget gives a weighted goal error of `4.640194561e-5`, compared
with `4.630657104e-5` for native FATROP lifted multiple shooting. Both runs hit
the iteration limit. IPPAS's final primal residual is approximately `0.259`;
this comparison establishes the requested goal-error target, not KKT convergence
of that contact trajectory. Run the reproducible comparison with:

```bash
JAX_PLATFORMS=cpu python -m mujodico.trajopt._src.ippas.push_sphere_2d_test
```
