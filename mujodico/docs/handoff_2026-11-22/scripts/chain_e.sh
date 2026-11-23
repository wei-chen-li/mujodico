#!/bin/bash
# new seeds: exact vs Gauss-Newton (all IP) vs Gauss-Newton in cycles only
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V13flags.txt)
for T in PushT2dHard:6:18 PushT2dHardLong:6:12; do
  t=${T%%:*}; r=${T#*:}
  $H/scripts/vplf.sh $t $r EXn seeds.jsonl $V | tail -1 | sed "s/^/$t EXn: /"
  $H/scripts/vplf.sh $t $r GNn seeds.jsonl $V --ip-hess 0 | tail -1 | sed "s/^/$t GNn: /"
  $H/scripts/vplf.sh $t $r GCn seeds.jsonl $V --ip-hess-cycle 0 | tail -1 | sed "s/^/$t GCn: /"
done
for L in GC4 GC5 GC6 GC7; do $H/scripts/vplf.sh AllegroPen 0:1 $L speed.jsonl $V --ip-hess-cycle 0 | tail -1 | sed "s/^/pen $L: /"; done
echo EDONE
