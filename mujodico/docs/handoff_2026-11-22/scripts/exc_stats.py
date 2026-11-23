"""Statistics of fidelity-cycle excursions and ladder phases from stored lfddp_v.py rows (no solver run).

usage: exc_stats.py <rows.jsonl> [more.jsonl ...] [--task PREFIX] [--gate-tau 0.05]
Rows with at least one fidelity cycle and no LPEC round are used (a row stores only its best round).
Prints: productive / unproductive excursions (by level, near misses), first true-law response, a retrospective
"plan-goal gate", the true-goal decrease per ladder phase, accepted step sizes, and the time split.
The gate numbers are a filter on logged streams, not a controlled run: stopping an excursion changes the path."""
import sys, json, math
from collections import Counter, defaultdict
import numpy as np

args = sys.argv[1:]
opt = lambda k, d, f=str: f(args[args.index(k) + 1]) if k in args else d
task = opt('--task', ''); tau = opt('--gate-tau', 0.05, float)
files = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or not args[i - 1].startswith('--'))]
rows = [json.loads(l) for f in files for l in open(f)]
rows = [r for r in rows if r['task'].startswith(task) and r.get('cycles', 0) > 0 and not r.get('lpec')]
print(f'{len(rows)} rows with cycles and no LPEC round; labels {dict(Counter(r["label"] for r in rows))}')

exc = []
for r in rows:   # match each cyc_log entry to its read-out stream (same start goal, level and stop reason, in order)
  ss, j = r['streams'], 0
  for c in r['cyc_log']:
    while j < len(ss):
      s = ss[j]; j += 1
      if s.get('r_start') is not None and abs(s['r_start'] - c['start']) < 1e-9 and abs(s['sig'] - c['sig']) < 1e-12 \
          and s['hand'] == c['hand']:
        exc.append(dict(st=s['stream'], r0=c['start'], its=c['its'], sig=c['sig'], hand=c['hand'],
                        gain=1 - c['after_ip'] / c['start']))
        break
P = [e for e in exc if e['gain'] > 0]; U = [e for e in exc if e['gain'] <= 0]
its = lambda S: sum(e['its'] for e in S)
print(f'excursions {len(exc)}: productive {len(P)} ({its(P)} its), unproductive {len(U)} ({its(U)} its); '
      f'unproductive stop reasons {dict(Counter(e["hand"] for e in U))}')
print('by level (productive, unproductive):',
      {s: (sum(e['sig'] == s for e in P), sum(e['sig'] == s for e in U)) for s in sorted({e['sig'] for e in exc}, reverse=True)})
print(f'incumbent below 0.03 at the start: productive {sum(e["r0"] < 0.03 for e in P)}, unproductive {sum(e["r0"] < 0.03 for e in U)}')
first = lambda e: next((it for it, p, rr in e['st'] if rr < e['r0'] - 1e-12), None)
f = np.array([first(e) for e in P if first(e) is not None])
if len(f):
  print('first true-law response of productive excursions (IP its): median %.0f, 80%% %.0f, 90%% %.0f, 95%% %.0f, max %.0f'
        % tuple(np.percentile(f, [50, 80, 90, 95, 100])))

def pdrop(e, k):   # relative drop of the plan goal within the first k iterations
  p0 = e['st'][0][1]
  return (p0 - min([p for it, p, rr in e['st'] if it <= k] or [p0])) / max(p0, 1e-12)
for name, S in (('productive', P), ('unproductive', U)):
  if S:
    print(f'plan-goal relative drop by it 20, {name}: quartiles', np.round(np.percentile([pdrop(e, 20) for e in S], [25, 50, 75]), 3))
for k in (20, 40):   # gate: at iteration k, no true response yet and plan-goal drop < tau -> stop
  stop = lambda e: e['its'] > k and pdrop(e, k) < tau and not any(rr < e['r0'] - 1e-12 for it, p, rr in e['st'] if it <= k)
  Pt = [e for e in P if e['gain'] >= tau]
  print(f'gate at it {k}: stops {sum(map(stop, U))}/{len(U)} unproductive, {sum(map(stop, Pt))}/{len(Pt)} tau-productive; '
        f'saves {sum(e["its"] - k for e in exc if stop(e))} of {its(exc)} excursion its')

seg = defaultdict(lambda: [0, 0.0]); alph = Counter(); T = defaultdict(float)
for r in rows:   # trace entries: (it, sig, boosts, J, r, r_inc, alpha, reg), one per read-out of the feasible phase
  prev = None; first_cyc = r['cyc_log'][0]['it']
  for t in r['trace']:
    it, sig, b, _, _, rinc, al = t[:7]; alph[al] += 1
    key = ('before the first cycle' if it <= first_cyc else 'after a cycle') + f', sig {sig:g}, b {b}'
    seg[key][0] += 1
    if prev is not None and rinc < prev:
      seg[key][1] += math.log(prev / max(rinc, 1e-12))
    prev = rinc
  for k_, v in r['timing'].items():
    T[k_] += v
  T['wall'] += r['time']
print('true-goal decrease by ladder phase (read-outs, sum of log decrease):')
for k_ in sorted(seg):
  print(f'  {k_:44s} {seg[k_][0]:6d} {seg[k_][1]:8.1f}')
print(f'  {"productive excursions":44s} {its(P):6d} {sum(-math.log(1 - e["gain"]) for e in P if e["gain"] < 1):8.1f}  (IP its)')
n = sum(alph.values())
print('accepted step sizes at read-outs:', {a: f'{c / n:.0%}' for a, c in sorted(alph.items(), key=lambda kv: -kv[0])})
print('time (s):', {k_: round(v) for k_, v in T.items()})
