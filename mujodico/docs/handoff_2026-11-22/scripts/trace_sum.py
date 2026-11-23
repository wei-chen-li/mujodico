"""Summarize a perfetto trace (jax.profiler, create_perfetto_trace=True): GPU busy time, kernel count, top kernels
by total time, grouped by XLA op name (fusion names stripped of numeric suffixes).  usage: trace_sum.py <dir> [its]"""
import sys, gzip, json, glob, re, collections
f = sorted(glob.glob(sys.argv[1] + '/**/*.trace.json.gz', recursive=True) + glob.glob(sys.argv[1] + '/**/perfetto_trace.json.gz', recursive=True))[-1]
its = int(sys.argv[2]) if len(sys.argv) > 2 else 1
d = json.load(gzip.open(f))
ev = d['traceEvents'] if isinstance(d, dict) else d
pname = {e['pid']: e['args']['name'] for e in ev if e.get('ph') == 'M' and e.get('name') == 'process_name'}
gpu = [e for e in ev if e.get('ph') == 'X' and 'GPU' in pname.get(e.get('pid'), '') or ('/device:GPU' in pname.get(e.get('pid'), ''))]
gpu = [e for e in gpu if e.get('ph') == 'X']
tot = sum(e['dur'] for e in gpu) / 1e3
span = (max(e['ts'] + e['dur'] for e in gpu) - min(e['ts'] for e in gpu)) / 1e3 if gpu else 0
print(f'{f}\nGPU events {len(gpu)} ({len(gpu)/its:.0f}/it), busy {tot:.0f} ms ({tot/its:.1f} ms/it), span {span:.0f} ms; processes: {set(pname.values())}')
by = collections.Counter(); cnt = collections.Counter()
for e in gpu:
  n = e.get('args', {}).get('hlo_op', e['name']); n = re.sub(r'[._]\d+$', '', n); n = re.sub(r'\.\d+', '', n)
  by[n] += e['dur'] / 1e3; cnt[n] += 1
for n, t in by.most_common(25):
  print(f'{t/its:8.2f} ms/it {cnt[n]/its:7.1f} calls/it  {n[:90]}')
