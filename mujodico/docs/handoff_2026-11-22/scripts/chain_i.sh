#!/bin/bash
# final evaluation of V14 = V13 + GN in cycle excursions + in-process LPEC fallback (6 rounds)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V14flags.txt)
export GRB_TL=300
$H/scripts/vplf.sh PushT2dHard 0:30 V14 final.jsonl $V | tail -1 | sed "s/^/PushT2dHard 0:30: /"
$H/scripts/vplf.sh PushT2d 0:18 V14 final.jsonl $V | tail -1 | sed "s/^/PushT2d 0:18: /"
$H/scripts/vplf.sh PushT2dHardLong 0:18 V14 final.jsonl $V | tail -1 | sed "s/^/PushT2dHardLong 0:18: /"
for L in V14p1 V14p2 V14p3 V14p4 V14p5; do $H/scripts/vplf.sh AllegroPen 0:1 $L final.jsonl $V | tail -1 | sed "s/^/pen $L: /"; done
$H/scripts/vpar.sh PushSphere2d 0:10 V14par final.jsonl 3 $V | sed "s/^/vpar PushSphere2d 0:10 k=3: /"
echo IDONE
