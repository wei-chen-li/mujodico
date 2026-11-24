"""E7 (HANDOFF_ladder_search.md §4.7): stop-class counts and the fixed-point test (restart from the returned plan under
deterministic ops; same plan = true goal within 1 % and controls within 1e-6).  usage: e7_stats.py <e7.jsonl> <plans dir>"""
import sys, json, collections
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; P = sys.argv[2]
base = {(r['task'], r['seed']): r for r in rows if r['label'] == 'E7'}
fp = {(r['task'], r['seed']): r for r in rows if r['label'] == 'E7fp'}
for task in sorted({k[0] for k in base}):
  rs = [r for k, r in base.items() if k[0] == task]
  print(task, 'solved', sum(r['true_ok'] for r in rs), '/', len(rs), dict(collections.Counter(r['stop_class'] for r in rs)))
print()
agg = collections.defaultdict(lambda: [0, 0, 0])
for k, r1 in sorted(fp.items()):
  r0 = base[k]; t, s = k
  u0 = np.load(f'{P}/E7_{t}_{s}.npz')['us']; u1 = np.load(f'{P}/E7fp_{t}_{s}.npz')['us']
  du = float(np.max(np.abs(u0 - u1))); same_g = abs(r1['true_goal'] - r0['true_goal']) <= 0.01 * max(r0['true_goal'], 1e-12)
  fixed = same_g and du <= 1e-6; impr = r1['true_goal'] < 0.99 * r0['true_goal']
  a = agg[r0['stop_class']]; a[0] += 1; a[1] += fixed; a[2] += impr
  print(f"{t:28s} {s:3d} {r0['stop_class']:14s} goal {r0['true_goal']:.4f} -> {r1['true_goal']:.4f}  max|du| {du:.1e}  "
        f"{'FIXED' if fixed else ('improved' if impr else 'moved')}  t {r1['time']:.0f}")
print()
for c, (n, f, i) in sorted(agg.items()):
  print(f"{c:14s} n {n:2d}  fixed points {f:2d}  improved {i:2d}")
