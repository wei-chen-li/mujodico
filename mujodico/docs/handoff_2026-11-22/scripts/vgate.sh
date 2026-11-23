#!/bin/bash
# usage: vgate.sh <label> <out.jsonl-name> [extra lfddp_v.py flags]  -- gate set (90) + knob (20), sequential
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts
L=$1; O=$2; shift 2
for TR in FingersSphere2dRandomizedHardLong:0:24 FingersSphere2dRandomized:0:24 PushSphere2d:0:10 PushSphere2dHard:0:10 AllegroBallTableHard:0:10 PushMultiSphere2d:0:6 PushBox2d:0:6 PushSquareKnobTable:0:20; do
  T=${TR%%:*}; R=${TR#*:}
  $H/vplf.sh $T $R $L $O "$@" | tail -1 | sed "s/^/$T: /"
done
echo GATEDONE
