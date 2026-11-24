"""Isolated replica of lfddp_v's CPU DDP backward pass (Riccati + _boxqp) at gripper vs Allegro sizes, random data.
Used 2026-11-24 to find the OpenBLAS oversubscription (see HANDOFF_bench 5.8).  EXTRA=<xla flags>; try OPENBLAS_NUM_THREADS=1."""
import os, sys, time
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16 ' + os.environ.get('EXTRA', '')
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jp, numpy as np
sys.path.insert(0, '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts')
from lfddp_v import _boxqp
def make(nx, nu):
  lb, ub = -jp.ones(nu), jp.ones(nu); I = jp.eye(nu)
  def backward(L, U, reg):
    A, B, a, lx, lu, lxx, lux, luu, VxN, VxxN = L
    def step(carry, inp):
      Vx, Vxx = carry
      A_, B_, a_, lx_, lu_, lxx_, lux_, luu_, u_ = inp
      Vx_ = Vx + Vxx @ a_
      Qx = lx_ + A_.T @ Vx_; Qu = lu_ + B_.T @ Vx_
      Qxx = lxx_ + A_.T @ Vxx @ A_; Quu = luu_ + B_.T @ Vxx @ B_; Qux = lux_ + B_.T @ Vxx @ A_
      Qr = 0.5 * (Quu + Quu.T) + reg * I
      ok = jp.all(jp.isfinite(jp.linalg.cholesky(Qr)))
      k, K = _boxqp(Qr, Qu, Qux, lb - u_, ub - u_)
      Vx_n = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
      Vxx_n = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
      return (Vx_n, 0.5 * (Vxx_n + Vxx_n.T)), (k, K, k @ Qu, ok)
    (_, _), (k, K, d1, ok) = jax.lax.scan(step, (VxN, VxxN), (A, B, a, lx, lu, lxx, lux, luu, U), reverse=True)
    return k, K, jp.sum(d1), jp.all(ok)
  return jax.jit(backward)
r = np.random.default_rng(0)
for name, nx, nu, N in [('gripper lift', 42, 7, 50), ('gripper stack', 42, 7, 90), ('allegro eraser', 69, 22, 120), ('allegro hammer', 59, 22, 150)]:
  f = make(nx, nu)
  A = 0.1 * r.standard_normal((N, nx, nx)) + np.eye(nx); B = 0.1 * r.standard_normal((N, nx, nu))
  spd = lambda n: (lambda M: M @ M.swapaxes(1, 2) / n + np.eye(n))(r.standard_normal((N, n, n)))
  L = (A, B, r.standard_normal((N, nx)), r.standard_normal((N, nx)), r.standard_normal((N, nu)), spd(nx),
       0.1 * r.standard_normal((N, nu, nx)), spd(nu), r.standard_normal(nx), np.eye(nx))
  L = tuple(jp.asarray(x) for x in L); U = jp.asarray(0.5 * r.standard_normal((N, nu)))
  jax.block_until_ready(f(L, U, 1e-6))
  ts = []
  for _ in range(10):
    t0 = time.perf_counter(); jax.block_until_ready(f(L, U, 1e-6)); ts.append(time.perf_counter() - t0)
  print(f'{name:15s} nx {nx} nu {nu} N {N}: bwd median {1e3*np.median(ts):.1f} ms  min {1e3*min(ts):.1f} ms', flush=True)
