"""PLF-V with an LPEC branch step when no smoothed model responds (outer driver; monotone acceptance on the true goal).

  plan 0: PLF-V from scratch (or --from <plan.npz>).
  round r (while the incumbent misses the goal):
    1. LPEC at the incumbent (lpec.py --quad 1, trust radius D): the branch step at the biactive contacts.
       No biactive index -> B-stationary -> stop.
    2. PLF-V from the step's true rollout (vplf.sh ... --from-us <step_roll.npz>): its first excursion starts there.
    3. Accept the re-solve if it lowers the incumbent's true goal (keep D); else D /= 2.5 (at most 2 shrinks in a row).
Each PLF-V call already contains the fidelity cycles; the LPEC only runs where every smoothed level has failed.
usage: pv_lpec.py <Task> <seed> <outdir> [--from plan.npz] [--rounds 4] [--delta 0.05]"""
import sys, os, json, time, shutil, subprocess
import numpy as np

H = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts'
PY = '/home/gchou/PycharmProjects/mujodico-main-dev/.venv/bin/python'
ROOT = '/home/gchou/PycharmProjects/mujodico-main-dev'
V = open(os.environ.get('VFLAGS', f'{H}/../results/vcycle/V13flags.txt')).read().split()   # PLF-V flags (tonight's runs: V11flags.txt)

name, seed, out = sys.argv[1], int(sys.argv[2]), os.path.abspath(sys.argv[3])
args = sys.argv[4:]
opt = lambda k, d, f=float: f(args[args.index(k) + 1]) if k in args else d
rounds, delta, start = opt('--rounds', 4, int), opt('--delta', 0.05), opt('--from', None, str)
os.makedirs(out, exist_ok=True)
env_cpu = dict(os.environ, JAX_PLATFORMS='cpu', GRB_TL='300')
PLANS = f'{H}/../results/vcycle/plans'


def plfv(label, extra):
  t0 = time.perf_counter()
  subprocess.run([f'{H}/vplf.sh', name, f'{seed}:{seed + 1}', label, 'pv_lpec.jsonl'] + V + extra, cwd=ROOT,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  p = f'{PLANS}/{label}_{name}_{seed}.npz'
  row = [json.loads(l) for l in open(f'{H}/../results/vcycle/pv_lpec.jsonl') if f'"label": "{label}"' in l][-1]
  return p, float(row['true_goal']), time.perf_counter() - t0


T0 = time.perf_counter()
tag = os.path.basename(out)
if start is None:
  inc, g_inc, t = plfv(f'{tag}_r0', [])
else:
  inc, g_inc, t = plfv(f'{tag}_r0', ['--from-us', os.path.abspath(start)])
rec = dict(task=name, seed=seed, goal0=g_inc, rounds=[])
print(f'{tag}: PLF-V {g_inc:.4f} ({t:.0f} s)', flush=True)
D, n_shrink, reason = delta, 0, 'rounds'
for r in range(1, rounds + 1):
  if g_inc < 0.01:
    reason = 'solved'; break
  pre = f'{out}/r{r}'
  t0 = time.perf_counter()
  subprocess.run([PY, '-u', f'{H}/lpec.py', name, inc, pre, '--quad', '1', '--deltas', str(D)], env=env_cpu, cwd=ROOT,
                 stdout=open(f'{pre}_lpec.log', 'w'), stderr=subprocess.STDOUT)
  try:
    summ = json.load(open(f'{pre}_summary.json'))
  except Exception:
    reason = 'lpec failed'; break
  info = dict(round=r, delta=D, nbi=summ['nbi'], t_lpec=time.perf_counter() - t0)
  if summ['nbi'] == 0:
    rec['rounds'].append(info); reason = 'B-stationary'; break
  if not summ['results']:
    rec['rounds'].append(info); reason = 'lpec: no step'; break
  p, g, t = plfv(f'{tag}_r{r}', ['--from-us', f'{pre}_D{D}_roll.npz'])
  info.update(pressed=len(summ['results'][0]['pressed']), step_goal=summ['results'][0]['goal_true'], goal=g, t_plfv=t,
              accepted=bool(g < g_inc - 1e-4))
  rec['rounds'].append(info)
  print(f'  round {r} (delta {D:g}): {summ["nbi"]} biactive, pressed {info["pressed"]}, step {info["step_goal"]:.3f} '
        f'-> PLF-V {g:.4f} ({t:.0f} s) {"accept" if info["accepted"] else "reject"}', flush=True)
  if info['accepted']:
    inc, g_inc, n_shrink = p, g, 0
  else:
    if n_shrink >= 2:
      reason = 'no improvement'; break
    D, n_shrink = D / 2.5, n_shrink + 1
else:
  reason = 'solved' if g_inc < 0.01 else 'rounds'
rec.update(final_goal=g_inc, ok=g_inc < 0.01, reason=reason, wall=time.perf_counter() - T0, final_plan=inc)
print(f'{tag}: final {g_inc:.4f} {reason}, wall {rec["wall"]:.0f} s', flush=True)
with open(f'{out}/../pv_lpec_summary.jsonl', 'a') as f:
  f.write(json.dumps(rec) + '\n')
