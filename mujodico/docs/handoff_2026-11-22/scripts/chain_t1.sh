#!/bin/bash
# waits for this session's pen jobs, then: JarC PLF vs PLF-V, timing pair PLF vs PLF-V (fingers-hard, fingers, PushBox2d)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts
while pgrep -f "lfddp_v.py AllegroPen" > /dev/null; do sleep 20; done
V="$(cat $H/../results/vcycle/V11flags.txt)"
SCRIPT=lfddp.py $H/vplf.sh AllegroJarC 0:5 PLFm3r plfv_hard.jsonl --clear-margin 0.003 | tail -1
$H/vplf.sh AllegroJarC 0:5 PV11r plfv_hard.jsonl $V --clear-margin 0.003 | tail -1
for T in FingersSphere2dRandomizedHardLong:0:24 FingersSphere2dRandomized:0:24 PushBox2d:0:6; do
  TT=${T%%:*}; R=${T#*:}
  SCRIPT=lfddp.py $H/vplf.sh $TT $R TPLF timing.jsonl | tail -1 | sed "s/^/PLF $TT: /"
  $H/vplf.sh $TT $R TV11 timing.jsonl $V | tail -1 | sed "s/^/V11 $TT: /"
done
echo CHAINDONE
