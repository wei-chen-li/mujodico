"""Cost of one IP read-out (as lfddp_v.readout_ip): lin (GPU or CPU), TVLQR (CPU), open + closed-loop true rollouts
(CPU, one vmapped call vs two CPU devices concurrently).  usage: ro_bench.py <Task>"""
import os, sys, time
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8').strip()
H_ = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs'
sys.path.insert(0, H_ + '/handoff_2026-11-22/scripts'); sys.path.insert(0, H_ + '/handoff_2026-11-14/followup/scripts')
import numpy as np, jax, jax.numpy as jp
jax.config.update('jax_enable_x64', True)
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting, util
import hard_tasks, extra_tasks  # noqa
from reloc_lms import LMS_KW
from lfddp_v import LFDDP
name = sys.argv[1]
task = make_task(name, 0)
lms = IppasLiftedMultipleShooting(task, smoothing_schedule=[dict(smoothing=1.0, iters=10, hess=[1, 0, 0])], max_iter=10, **LMS_KW)
truem = IppasLiftedMultipleShooting(task); truem.rollout(task.x_0, np.zeros((task.horizon, task.nu)), 0.0)
S = LFDDP(task, lms); N, nu, nx = S.N, S.nu, S.nx
CPUS = jax.devices('cpu'); CPU = CPUS[0]; GPU = jax.devices()[0]
toC = lambda t: jax.device_put(t, CPU)
_ts = truem.transcription.true_step; qidx = list(task.quaternion_w_idxs)
_sc = lambda Q, g, x: jp.where(jp.all(jp.isfinite(x)), util.state_cost(Q, g, x, qidx), jp.inf)
def _true_one(s, Q, g, Xp, Up, K):
  def body(x, k):
    u = jp.clip(Up[k] + s * (K[k] @ (x - Xp[k])), S.lb, S.ub)
    xn, _ = _ts(x, u, jp.asarray(0.0)); return xn, u
  xN, Ua = jax.lax.scan(body, Xp[0], jp.arange(Up.shape[0])); return _sc(Q, g, xN), Ua
true_both = jax.jit(jax.vmap(_true_one, in_axes=(0, None, None, None, None, None))); true_one = jax.jit(_true_one)
Qm = jp.asarray(getattr(task, 'metric_Q', task.Q)); g0 = jp.asarray(task.x_goal)
U = 0.1 * jp.ones((N, nu)); X, C = S._roll(jp.asarray(task.x_0), U, jp.asarray(S.base))
nq_ = task.model.nq
Qd = jp.diag(jp.concatenate([jp.ones(nq_), 0.1 * jp.ones(nx - nq_)])); R = 1e-2 * jp.eye(nu)
P = lms.transcription.params
def tm(f, n=5):
  f(); t = time.perf_counter()
  for _ in range(n): f()
  return 1000 * (time.perf_counter() - t) / n
for dev, dn in ((GPU, 'gpu'), (CPU, 'cpu')):
  a = [jax.device_put(t, dev) for t in (X, U, C, jp.asarray(S.base), P, jp.zeros((N, nx)), jp.asarray(1e-3))]
  print(f'{name}: lin on {dn}: {tm(lambda: jax.block_until_ready(S._lin(*a))):.1f} ms', flush=True)
L = S._lin(*[toC(t) for t in (X, U, C, jp.asarray(S.base), P, jp.zeros((N, nx)), jp.asarray(1e-3))])
print(f'{name}: tvlqr cpu: {tm(lambda: jax.block_until_ready(S._tvlqr(L[0], L[1], toC(Qd), toC(R), toC(10 * Qd)))):.1f} ms', flush=True)
K = S._tvlqr(L[0], L[1], toC(Qd), toC(R), toC(10 * Qd))
args = [toC(t) for t in (Qm, g0, X, U, K)]
print(f'{name}: true_both (vmapped, 1 call): {tm(lambda: jax.block_until_ready(true_both(toC(jp.asarray([0.0, 1.0])), *args))):.1f} ms', flush=True)
print(f'{name}: true_one x1: {tm(lambda: jax.block_until_ready(true_one(toC(jp.asarray(1.0)), *args))):.1f} ms', flush=True)
def two():
  o = [true_one(*[jax.device_put(t, CPUS[5 + i]) for t in [jp.asarray(float(i))] + args]) for i in range(2)]
  jax.block_until_ready(o)
print(f'{name}: true_one on 2 CPU devices concurrently: {tm(two):.1f} ms', flush=True)
