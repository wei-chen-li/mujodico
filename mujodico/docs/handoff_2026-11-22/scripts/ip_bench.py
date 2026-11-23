"""Per-iteration cost of the lifted IP (resumable homotopy loop, chunks of 5) from the zero-control start of a task,
as PLF's first exploration runs it.  usage: ip_bench.py <Task> <hess 0|1> [iters 60] [--profile dir]"""
import sys, time
H_ = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs'
sys.path.insert(0, H_ + '/handoff_2026-11-22/scripts'); sys.path.insert(0, H_ + '/handoff_2026-11-14/followup/scripts')
import numpy as np, jax, jax.numpy as jp
jax.config.update('jax_enable_x64', True)
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting
import hard_tasks, extra_tasks  # noqa
from reloc_lms import LMS_KW
from lfddp import LFDDP
from hs_export import ExportedHomotopyIpPas
name, hess = sys.argv[1], float(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('-') else 60
prof = sys.argv[sys.argv.index('--profile') + 1] if '--profile' in sys.argv else None
mu = (0.1, 0.1) if name.startswith(('PushT', 'PushBox')) else (1e-3, 1e-2)
task = make_task(name, 0)
PH = dict(smoothing=1.0, iters=10, hess=[hess, 0, 0], mu_init=mu[0], mu_min=mu[1])
lms = IppasLiftedMultipleShooting(task, smoothing_schedule=[PH], max_iter=400, **LMS_KW)
S = LFDDP(task, lms); N, nu = S.N, S.nu
X, C = S._roll(jp.asarray(task.x_0), jp.zeros((N, nu)), jp.asarray(S.base))
prm = lms._params(PH); prm['hom'] = jp.asarray([1.0, 1.0, 0.0, 1.0, S.base])
HS = ExportedHomotopyIpPas(lms.solver)
Sip = HS.start(prm, X, jp.concatenate([jp.zeros((N, nu)), C], 1), 5, max_iter=400); jax.block_until_ready(Sip.cur.x)
Sip = HS.resume(prm, Sip, 10); jax.block_until_ready(Sip.cur.x)
it0 = int(Sip.it); t0 = time.perf_counter()
if prof: jax.profiler.start_trace(prof, create_perfetto_trace=True)
while int(Sip.it) < it0 + iters and HS.paused(Sip):
  Sip = HS.resume(prm, Sip, int(Sip.it) + 5)
jax.block_until_ready(Sip.cur.x)
if prof: jax.profiler.stop_trace()
dt = time.perf_counter() - t0; n = int(Sip.it) - it0
st = np.asarray(Sip.stats)
print(f'{name} hess {hess:g}: {n} its in {dt:.2f} s = {1000 * dt / max(n, 1):.1f} ms/it; nx {task.nx} nu {nu} nw {lms.transcription.nw} N {N}; stats {st.tolist()}', flush=True)
