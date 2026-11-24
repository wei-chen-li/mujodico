"""E1: compare rows of lfddp_v.py (labels E1v*) and lfddp_l.py (E1l*) field by field.  usage: e1_cmp.py <rows.jsonl>"""
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1])]
d = {}
for r in rows:
  side = r['label'][2]; key = (r['task'], r['seed'], r['label'][3:])
  d.setdefault(key, {})[side] = r
fields = ('true_goal', 'it', 'ip_its', 'reason', 'events', 'cyc_log', 'lpec', 'cycles', 'fine_cycles')
nok = 0
for k, v in sorted(d.items()):
  if 'v' not in v or 'l' not in v:
    print(k, 'incomplete'); continue
  diff = [f for f in fields if v['v'].get(f) != v['l'].get(f)]
  nok += not diff
  print(k, 'IDENTICAL' if not diff else 'DIFF ' + str(diff), round(v['v']['true_goal'], 6), round(v['l']['true_goal'], 6))
print('identical', nok, 'of', len(d))
