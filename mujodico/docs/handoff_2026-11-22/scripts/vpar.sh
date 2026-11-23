#!/bin/bash
# usage: vpar.sh <Task> <a:b> <label> <out.jsonl-name> <k> [extra lfddp_v.py flags]
# vplf.sh with the seed range split over k concurrent processes (throughput for small, launch-bound tasks:
# PushT2dHard 12 seeds, 3 GPU processes 1.65x; CPU-only processes: CPU=1). Rows are appended to <out> per part.
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
T=$1; R=$2; L=$3; O=$4; K=$5; shift 5
a=${R%%:*}; b=${R#*:}; n=$((b - a)); step=$(((n + K - 1) / K))
[ -n "$CPU" ] && export JAX_PLATFORMS=cpu
for ((s = a; s < b; s += step)); do
  e=$((s + step < b ? s + step : b))
  LOGSUF=_p$s $H/scripts/vplf.sh $T $s:$e $L $O.p$s "$@" > /dev/null &
done
wait
for ((s = a; s < b; s += step)); do cat $H/results/vcycle/$O.p$s >> $H/results/vcycle/$O && rm $H/results/vcycle/$O.p$s; done
grep -h '"label": "'$L'"' $H/results/vcycle/$O | python3 -c "
import sys, json
rs = [json.loads(l) for l in sys.stdin]; rs = [r for r in rs if r['task'] == '$T']
print('solved', sum(r['true_ok'] for r in rs), '/', len(rs))"
