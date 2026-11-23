"""Cost of LFDDP's line search (5 step sizes) on CPU devices, as lfddp_v.py runs it, and alternatives.
usage: ls_bench.py <Task>   (set XLA_FLAGS before to test CPU threading options)"""
import os, sys, time, threading
if '--xla_force_host_platform_device_count' not in os.environ.get('XLA_FLAGS', ''):
  os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8').strip()
H_ = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs'
sys.path.insert(0, H_ + '/handoff_2026-11-22/scripts'); sys.path.insert(0, H_ + '/handoff_2026-11-14/followup/scripts')
import numpy as np, jax, jax.numpy as jp
jax.config.update('jax_enable_x64', True)
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting
import hard_tasks, extra_tasks  # noqa
from reloc_lms import LMS_KW
from lfddp_v import LFDDP
name = sys.argv[1]
task = make_task(name, 0)
lms = IppasLiftedMultipleShooting(task, smoothing_schedule=[dict(smoothing=1.0, iters=10, hess=[1, 0, 0])], max_iter=10, **LMS_KW)
S = LFDDP(task, lms); N, nu, nx = S.N, S.nu, S.nx
CPUS = jax.devices('cpu'); C0 = CPUS[0]
put = lambda t, d=C0: jax.device_put(t, d)
eps = put(jp.asarray(S.base * 1e-2)); P = put(lms.transcription.params); mu = put(jp.asarray(1e-3)); x0 = put(jp.asarray(task.x_0))
U = put(0.05 * jp.ones((N, nu))); X, C = S._roll(x0, U, eps)
L = S._lin(X, U, C, eps, P, put(jp.zeros((N, nx))), mu)
k, K, *_ = S._bwd(L, U, put(jp.asarray(1e-3)))
fbar, hbar = L[8], L[9]
ALPHAS = (1.0, 0.5, 0.25, 0.1, 0.03)
ks = put(jp.asarray(0.0))
def one(dev, al):
  return S._fwd(*[put(t, dev) for t in (X, U, C, k, K, fbar, hbar)], put(jp.asarray(al), dev), *[put(t, dev) for t in (eps, x0, P, mu, ks)])
def par_devices():   # what lfddp_v.py does (ls_mode 'par')
  outs = [one(CPUS[i], al) for i, al in enumerate(ALPHAS)]
  return [float(o[3]) for o in outs]
def single():
  return float(one(C0, 1.0)[3])
def seq():
  return [float(one(C0, al)[3]) for al in ALPHAS]
def threads():
  res = [None] * 5
  def w(i, al):
    res[i] = float(one(CPUS[i], al)[3])
  ts = [threading.Thread(target=w, args=(i, al)) for i, al in enumerate(ALPHAS)]
  [t.start() for t in ts]; [t.join() for t in ts]
  return res
def batch():
  o = S._fwd_b(X, U, C, k, K, fbar, hbar, put(jp.asarray(ALPHAS)), eps, x0, P, mu, ks)
  return list(np.asarray(o[3]))
def tm(f, n=8):
  f(); f()
  t = time.perf_counter()
  for _ in range(n): r = f()
  return 1000 * (time.perf_counter() - t) / n, r
base = None
for nm, f in (('single rollout', single), ('5 seq, 1 device', seq), ('5 on 5 devices (current)', par_devices),
              ('5 on 5 devices, python threads', threads), ('vmapped batch, 1 device', batch)):
  ms, r = tm(f)
  print(f'{name}: {nm:34s} {ms:7.1f} ms   J[0] {np.atleast_1d(r)[0]:.12g}', flush=True)
