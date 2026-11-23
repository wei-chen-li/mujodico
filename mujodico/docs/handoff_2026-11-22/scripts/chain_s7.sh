#!/bin/bash
# option 7 (package): SPD-solve derivative by inverse products, gradient-stopped SAP factor; V14 flags
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
for L in S7p1 S7p2 S7p3; do $H/scripts/vplf.sh AllegroPen 0:1 $L speed3.jsonl $V | tail -1 | sed "s/^/pen $L: /"; done
$H/scripts/vplf.sh AllegroBallTableHard 0:10 S7 speed3.jsonl $V | tail -1 | sed "s/^/AllegroBallTableHard: /"
$H/scripts/vplf.sh AllegroJar 0:5 S7 speed3.jsonl $V | tail -1 | sed "s/^/AllegroJar: /"
$H/scripts/vplf.sh AllegroJarC 0:5 S7 speed3.jsonl $V --clear-margin 0.003 | tail -1 | sed "s/^/AllegroJarC: /"
echo S7DONE
