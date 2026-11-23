#!/bin/bash
# more new seeds: exact (V13) vs GN in cycles only
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V13flags.txt)
while pgrep -f "chain_g[.]sh" >/dev/null; do sleep 5; done
for T in PushT2dHard:18:30 PushT2d:6:18 PushT2dHardLong:12:18; do
  t=${T%%:*}; r=${T#*:}
  $H/scripts/vplf.sh $t $r EXn seeds.jsonl $V | tail -1 | sed "s/^/$t EXn: /"
  $H/scripts/vplf.sh $t $r GCn seeds.jsonl $V --ip-hess-cycle 0 | tail -1 | sed "s/^/$t GCn: /"
done
echo HDONE
