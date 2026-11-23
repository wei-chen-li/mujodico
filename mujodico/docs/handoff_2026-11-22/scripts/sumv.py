"""Summarize lfddp_v.py / c1_vcycle.py rows: sumv.py <log-or-jsonl> [label]"""
import sys, json
f = sys.argv[1]; lab = sys.argv[2] if len(sys.argv) > 2 else None
n = ok = 0; ts = []
for l in open(f):
    if not l.startswith('{'): continue
    r = json.loads(l)
    if lab and r['label'] != lab: continue
    n += 1; ok += r['true_ok']; ts.append(r['time'])
    if 'cyc_log' in r and r['cyc_log'] and 'sig' in r['cyc_log'][0]:
        cl = ' '.join(f"[s{c['sig']:g} {c['its']}{c['hand'][0]}: {c['start']:.3f}->{c['after_ip']:.3f}]" for c in r['cyc_log'])
        its = r['it'] + r['ip_its']
    else:
        cl = ' '.join(f"[L{c['level']:g}{'n' if c.get('near') else ''}: {c['start']:.3f}->{c['end']:.4f}]" for c in r.get('cyc_log', []))
        its = r['it']
    print(r['label'], r['task'], r['seed'], 'OK  ' if r['true_ok'] else 'miss', f"{r['true_goal']:.4f}", r.get('reason', ''), 'cyc', r.get('cycles'), cl, 'its', its, 't', round(r['time'], 1))
ts.sort()
print('solved', ok, '/', n, ' median t', round(ts[len(ts)//2], 1) if ts else None)
