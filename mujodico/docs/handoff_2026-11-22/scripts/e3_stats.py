"""E3 (HANDOFF_ladder_search.md): --lm 1 (step-size table) against --lm 2 (gain ratio), from rows with logged steps.
usage: e3_stats.py <rows.jsonl>"""
import sys, json, collections
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows:
  lab = r['label']
  arm = 'lm%d' % r['cfg']['lm']
  run = 'pen' if r['task'] == 'AllegroPen' else lab
  g[(r['task'], arm, run)].append(r)
print(f"{'task':34s} {'arm':4s} {'run':4s} {'solved':>7s} {'it med':>7s} {'ls s/inst':>9s} {'t med':>6s} | alpha=1 / .5 / .25 / .1 / .03 | reg med / max | rho med (lm2)")
for (task, arm, run), rs in sorted(g.items()):
  steps = [s for r in rs for s in (r.get('steps') or [])]
  al = collections.Counter(s[0] for s in steps); n = max(1, sum(al.values()))
  regs = np.array([s[2] for s in steps]) if steps else np.array([np.nan])
  rhos = np.array([s[1] for s in steps if s[1] is not None]) if steps else np.array([])
  times = [r['time'] for r in rs]
  tt = times[1:] if len(times) > 2 else times
  print(f"{task:34s} {arm:4s} {run:4s} {sum(r['true_ok'] for r in rs):3d}/{len(rs):<3d} {np.median([r['it'] for r in rs]):7.0f} "
        f"{np.mean([r['timing']['fwd'] for r in rs]):9.2f} {np.median(tt):6.1f} | "
        + ' / '.join(f"{100 * al[a] / n:3.0f}%" for a in (1.0, 0.5, 0.25, 0.1, 0.03))
        + f" | {np.nanmedian(regs):.1e} / {np.nanmax(regs):.1e} | {np.median(rhos) if rhos.size else float('nan'):.2f}")
