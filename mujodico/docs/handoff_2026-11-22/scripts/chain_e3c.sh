#!/bin/bash
# E3 rerun of the gain-ratio arm (the first E3 'lm2' arm ran --lm 1: first-occurrence flag parsing)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
while pgrep -f "chain_none_xyz[.]sh" >/dev/null; do sleep 10; done
for run in 1 2; do
  $H/scripts/lplf.sh FingersSphere2dRandomizedHardLong 0:24 E3Clm2r$run e3c.jsonl $V --lm 2 --log-steps 1 | tail -1 | sed "s/^/fingers-hard lm2 run$run: /"
  $H/scripts/lplf.sh PushT2dHard 0:12 E3Clm2r$run e3c.jsonl $V --lm 2 --log-steps 1 | tail -1 | sed "s/^/PushT2dHard lm2 run$run: /"
done
$H/scripts/lplf.sh AllegroPen 0:1 E3Clm2p e3c.jsonl $V --lm 2 --log-steps 1 | tail -1 | sed "s/^/pen lm2: /"
echo E3CDONE
