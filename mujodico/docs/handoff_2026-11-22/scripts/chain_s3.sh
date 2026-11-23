#!/bin/bash
# option 3: throughput of mixed GPU + CPU processes, PushT2dHard seeds 6:18, V14 flags (after chain_s7)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
while pgrep -f "chain_s7[.]sh" >/dev/null; do sleep 5; done
# warm both caches (GPU and CPU) for the current code
$H/scripts/vplf.sh PushT2dHard 6:7 WARMG par2.jsonl $V > /dev/null
JAX_PLATFORMS=cpu $H/scripts/vplf.sh PushT2dHard 6:7 WARMC par2.jsonl $V > /dev/null
run() {   # label, then pairs "dev:a:b"
  L=$1; shift; t0=$(date +%s.%N)
  for spec in "$@"; do d=${spec%%:*}; r=${spec#*:}
    if [ $d = c ]; then JAX_PLATFORMS=cpu LOGSUF=_${r/:/_} $H/scripts/vplf.sh PushT2dHard $r $L par2.jsonl $V > /dev/null &
    else LOGSUF=_${r/:/_} $H/scripts/vplf.sh PushT2dHard $r $L par2.jsonl $V > /dev/null & fi
  done; wait; t1=$(date +%s.%N)
  echo "$L: $(echo "$t1 - $t0" | bc) s, solved $(grep -h "\"label\": \"$L\"" $H/results/vcycle/par2.jsonl | grep -c '"true_ok": true')/12"
}
run P1G g:6:18
run P3G g:6:10 g:10:14 g:14:18
run P3G3C g:6:8 g:8:10 g:10:12 c:12:14 c:14:16 c:16:18
run P4G2C g:6:8 g:8:10 g:10:12 g:12:14 c:14:16 c:16:18
echo S3DONE
