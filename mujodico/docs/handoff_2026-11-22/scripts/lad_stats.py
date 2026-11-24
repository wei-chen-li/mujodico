"""Per (label, task): solved, median iterations (LFDDP + IP), median and total time, stop classes.  Pen labels <lab>p<k>
are pooled.  usage: lad_stats.py <rows.jsonl> [--by-task]"""
import sys, json, collections, re
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
g = collections.defaultdict(list)
for r in rows:
  lab = re.sub(r'p\d$', '', r['label']) if r['task'] == 'AllegroPen' else r['label']
  g[(r['task'], lab)].append(r)
print(f"{'task':36s} {'label':10s} {'solved':>8s} {'its med':>8s} {'t med':>7s} {'t sum':>8s} {'lpec rnds':>9s}  stop classes")
for (task, lab), rs in sorted(g.items()):
  its = [r['it'] + r['ip_its'] for r in rs]; ts = [r['time'] for r in rs]
  sc = collections.Counter(r.get('stop_class', '?') for r in rs)
  nl = sum(len(r.get('lpec') or []) for r in rs)
  print(f"{task:36s} {lab:10s} {sum(r['true_ok'] for r in rs):3d}/{len(rs):<4d} {np.median(its):8.0f} {np.median(ts):7.1f} {np.sum(ts):8.0f} {nl:9d}  "
        + ', '.join(f'{k} {v}' for k, v in sorted(sc.items())))
