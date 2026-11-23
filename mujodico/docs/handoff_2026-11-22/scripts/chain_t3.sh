#!/bin/bash
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts
V="$(cat $H/../results/vcycle/V11flags.txt)"
for T in PushT2d PushT2dHard; do
  SCRIPT=lfddp.py $H/vplf.sh $T 0:6 IPLF idle.jsonl | tail -1 | sed "s/^/PLF $T: /"
  $H/vplf.sh $T 0:6 IV11 idle.jsonl $V | tail -1 | sed "s/^/V11 $T: /"
done
$H/vplf.sh AllegroPen 0:1 IV11 idle.jsonl $V | tail -1 | sed "s/^/V11 pen: /"
echo CHAINDONE
