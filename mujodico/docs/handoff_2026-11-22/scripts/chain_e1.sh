#!/bin/bash
# E1 (HANDOFF_ladder_search.md): lfddp_v.py vs restructured lfddp_l.py, V14 flags, deterministic ops
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
P=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300 DET=1
for sc in v l; do
  export SCRIPT=$([ $sc = l ] && echo lfddp_l_e1.py || echo lfddp_v.py)
  $H/scripts/vplf.sh PushT2dHard 0:6 E1$sc e1.jsonl $V | tail -1 | sed "s/^/$sc PushT2dHard: /"
  $H/scripts/vplf.sh PushBox2d 0:6 E1$sc e1.jsonl $V | tail -1 | sed "s/^/$sc PushBox2d: /"
  $H/scripts/vplf.sh FingersSphere2dRandomizedHardLong 0:6 E1$sc e1.jsonl $V | tail -1 | sed "s/^/$sc fingers-hard: /"
  $H/scripts/vplf.sh PushSquareKnobTable 8:12 E1$sc e1.jsonl $V | tail -1 | sed "s/^/$sc knob: /"
  $H/scripts/vplf.sh PushT2dHard 25:26 E1${sc}lp e1.jsonl $V --from-us $P/GCn_PushT2dHard_{seed}.npz | tail -1 | sed "s/^/$sc lpec T25: /"
  $H/scripts/vplf.sh AllegroPen 0:1 E1$sc e1.jsonl $V | tail -1 | sed "s/^/$sc pen: /"
done
echo E1DONE
