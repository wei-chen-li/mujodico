#!/bin/bash
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts
V="$(cat $H/../results/vcycle/V11flags.txt)"
$H/vplf.sh PushT2dHard 0:6 PV11g plfv.jsonl $V | tail -1 | sed "s/^/PV11g T2dHard: /"
for i in 1 2; do
  SCRIPT=lfddp.py $H/vplf.sh PushBox2d 0:6 TPLF$i timing.jsonl | tail -1 | sed "s/^/PLF box $i: /"
  $H/vplf.sh PushBox2d 0:6 TV11$i timing.jsonl $V | tail -1 | sed "s/^/V11 box $i: /"
done
echo CHAINDONE
