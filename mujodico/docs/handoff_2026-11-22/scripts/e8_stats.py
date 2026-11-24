"""E8 (HANDOFF_ladder_search.md): one sigma0 excursion from each stored incumbent with the IP and with the lifted
(carried-gap) engine.  Productive: the excursion lowered the true goal; gain: r_start - r_end; first: iterations to the
first true-law response; ms/it of the excursion.  usage: e8_stats.py <e8.jsonl>"""
import sys, json, collections, os
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in rows:
  eng = r['label'].replace('E8x_', '')
  by[r['cfg']['from_us']][eng] = r
def stat(r):
  r0 = next(e[1] for e in r['events'] if e[0] == 'from-us')
  st = (r.get('streams') or [{}])[-1].get('stream', [])
  first = next((k for k, p, rr in st if rr < r0 - 1e-4), None)
  return r0, r['true_goal'], first, 1e3 * r['timing']['ip'] / max(1, r['ip_its']), r['ip_its'], r['handoff']
engs = ['ip', 'lifted', 'lifted_lam', 'lifted_filter', 'lifted_ff']
agg = collections.defaultdict(lambda: collections.defaultdict(list))
print(f"{'incumbent':44s} {'r0':>7s} | " + ' | '.join(f"{e:>10s} r_end  its ms/it hand" for e in engs))
for f, d in sorted(by.items()):
  name = os.path.basename(f).replace('.npz', '')
  grp = 'pen' if 'AllegroPen' in f else 'T'
  line = f"{name:44s} "
  for e in engs:
    if e not in d:
      line += ' | ' + ' ' * 30; continue
    r0, re, first, ms, its, hand = stat(d[e])
    if e == 'ip':
      line += f"{r0:7.4f}"
    line += f" | {re:7.4f} {its:4d} {ms:5.1f} {hand[:10]:10s}"
    a = agg[(grp, e)]
    a['prod'].append(re < r0 - 1e-4)   # r0 is logged to 4 decimals
    a['gain'].append(r0 - re)
    a['ms'].append(ms); a['its'].append(its)
    a['first'].append(first)
  print(line)
print()
for (grp, e), a in sorted(agg.items()):
  fr = [x for x in a['first'] if x is not None]
  print(f"{grp:4s} {e:10s} n {len(a['prod']):3d}  productive {100 * np.mean(a['prod']):5.1f}%  gain med {np.median(a['gain']):.4f} "
        f"mean {np.mean(a['gain']):.4f}  first-response its med {np.median(fr) if fr else float('nan'):.0f}  "
        f"its med {np.median(a['its']):.0f}  ms/it med {np.median(a['ms']):.1f}")
