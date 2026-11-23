"""Per run label: seeds solved before the first fidelity cycle (= what PLF alone returned in that process) vs after.
usage: basevsfinal.py <rows.jsonl> [task]"""
import sys, json, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
task = sys.argv[2] if len(sys.argv) > 2 else None
by = collections.OrderedDict()
for r in rows:
    if task and r['task'] != task: continue
    by.setdefault((r['label'], r['task']), []).append(r)
for (lab, t), rs in by.items():
    base = [(r['cyc_log'][0]['start'] if r.get('cyc_log') else r['true_goal']) for r in rs]
    fin = [r['true_goal'] for r in rs]
    tt = sorted(r['time'] for r in rs); its = sorted(r['it'] + r['ip_its'] for r in rs)
    print(f"{lab:8s} {t:14s} n={len(rs)} base {sum(b < 0.01 for b in base)} -> final {sum(f < 0.01 for f in fin)}  "
          f"median t {tt[len(tt)//2]:.1f} max t {tt[-1]:.1f} max its {its[-1]}  misses {[(r['seed'], round(r['true_goal'],3)) for r in rs if r['true_goal'] >= 0.01]}")
