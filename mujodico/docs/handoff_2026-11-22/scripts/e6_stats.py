"""E6 (HANDOFF_ladder_search.md): replay per-round cycle caps c on the logged LPEC rounds.  A round capped at c cycles
returns the incumbent at the start of cycle c+1 (or its final goal if it ran fewer); it is accepted if that is below
(1 - tau) r_ref.  Cost: iterations up to that point.  usage: e6_stats.py <e6.jsonl>"""
import sys, json
import numpy as np
TAU = 0.05; CAPS = (0, 1, 2, 4, 99)
rows = [json.loads(l) for l in open(sys.argv[1])]
tot = {c: [0, 0, 0] for c in CAPS}   # accepted rounds, same decision as full, iterations
for r in rows:
  print(f"{r['label']:10s} {r['task']:28s} {r['seed']:3d} {'OK ' if r['true_ok'] else 'miss'} goal {r['true_goal']:.4f} t {r['time']:6.1f} rounds {len(r.get('lpec') or [])}")
  for e in r.get('lpec') or []:
    if 'cyc_log' not in e:
      print('    ', e.get('stop', e.get('reason'))); continue
    cl = e['cyc_log']; rr = e['r_ref']; ip0 = e['ip_its'] - sum(c['its'] for c in cl)
    full_acc = e['accepted']; line = f"    r_ref {rr:.4f} goal {e['goal']:.4f} acc {int(full_acc)} cycles {len(cl)} its {e['it'] + e['ip_its']:5d} |"
    for c in CAPS:
      if c < len(cl):
        rc = cl[c]['start']; its = cl[c]['it'] + ip0 + sum(x['its'] for x in cl[:c])
      else:
        rc = e['goal']; its = e['it'] + e['ip_its']
      acc = rc < (1 - TAU) * rr
      tot[c][0] += acc; tot[c][1] += acc == full_acc; tot[c][2] += its
      line += f" c{c}: {rc:.4f}{'*' if acc else ' '} {its:5d}"
    print(line)
print()
for c in CAPS:
  print(f"cap {c:2d}: accepted {tot[c][0]:3d}, same decision as uncapped {tot[c][1]:3d}, iterations {tot[c][2]}")
