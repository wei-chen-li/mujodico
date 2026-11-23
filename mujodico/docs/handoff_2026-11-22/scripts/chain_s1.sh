#!/bin/bash
# speed option 1: single-threaded Eigen on CPU devices (CPU1T=1), V14 flags, vs final.jsonl V14 rows
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
CPU1T=1 $H/scripts/vplf.sh PushT2dHard 0:6 S1T speed2.jsonl $V | tail -1
$H/scripts/vplf.sh PushT2dHard 0:6 S1B speed2.jsonl $V | tail -1
for L in S1Tp1 S1Tp2 S1Tp3; do CPU1T=1 $H/scripts/vplf.sh AllegroPen 0:1 $L speed2.jsonl $V | tail -1; done
for L in S1Bp1 S1Bp2; do $H/scripts/vplf.sh AllegroPen 0:1 $L speed2.jsonl $V | tail -1; done
echo S1DONE
