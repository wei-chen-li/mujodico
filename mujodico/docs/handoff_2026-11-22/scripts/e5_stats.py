"""E5 (HANDOFF_ladder_search.md): --ladder auto against the per-task V14 rows.  Per task and run: solved, the responding
rung of the first exploration (auto-rung events: 0 = (sig 1, mu 1e-3), 1 = (1, 1e-1), 2 = (10, 1e-1)), auto-climbs,
IP iterations before the response, median iterations and seconds; reference: gate_gn GN1 (V14 gate) and final V14 (T).
usage: e5_stats.py <e5.jsonl> <vcycle dir>"""
import sys, json, collections
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; V = sys.argv[2]
ref = collections.defaultdict(list)
for f, lab in (('gate_gn.jsonl', 'GN1'), ('final.jsonl', 'V14')):
  for l in open(f'{V}/{f}'):
    r = json.loads(l)
    if r['label'] == lab and not (lab == 'V14' and r['seed'] >= 12):
      ref[r['task']].append(r)
g = collections.defaultdict(list)
for r in rows:
  g[(r['task'], r['label'])].append(r)
print(f"{'task':34s} {'run':5s} {'solved':>7s} {'rungs 0/1/2':>12s} {'climb':>5s} {'its->resp':>9s} {'its med':>7s} {'t med':>6s} | {'V14 solved':>10s} {'its med':>7s} {'t med':>6s}")
for (task, lab), rs in sorted(g.items()):
  rung = collections.Counter(); climbs = 0; pre = []
  for r in rs:
    ar = [e for e in r['events'] if e[0] == 'auto-rung']
    if ar:
      rung[ar[-1][1]] += 1; pre.append(sum(e[4] for e in ar))
    climbs += sum(1 for e in r['events'] if e[0] == 'auto-climb')
  rf = ref.get(task, [])
  tt = [r['time'] for r in rs]; tt = tt[1:] if len(tt) > 2 else tt     # the first seed of a process pays start-up
  rt = [r['time'] for r in rf]; rt = rt[1:] if len(rt) > 2 else rt
  print(f"{task:34s} {lab:5s} {sum(r['true_ok'] for r in rs):3d}/{len(rs):<3d} {'/'.join(str(rung[i]) for i in range(3)):>12s} {climbs:5d} "
        f"{np.median(pre) if pre else float('nan'):9.0f} {np.median([r['it'] + r['ip_its'] for r in rs]):7.0f} {np.median(tt):6.1f} | "
        + (f"{sum(r['true_ok'] for r in rf):5d}/{len(rf):<4d} {np.median([r['it'] + r['ip_its'] for r in rf]):7.0f} {np.median(rt):6.1f}" if rf else '   —'))
