#!/bin/bash
# ladder-search handoff E3 (gain-ratio regularization) + reference check (current lfddp_v defaults vs committed 44ca93a)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
while pgrep -f "chain_lv[.]sh" >/dev/null; do sleep 10; done
for run in 1 2; do
  for lmv in 1 2; do
    $H/scripts/lplf.sh FingersSphere2dRandomizedHardLong 0:24 E3lm${lmv}r$run e3.jsonl $V --lm $lmv --log-steps 1 | tail -1 | sed "s/^/fingers-hard lm$lmv run$run: /"
    $H/scripts/lplf.sh PushT2dHard 0:12 E3lm${lmv}r$run e3.jsonl $V --lm $lmv --log-steps 1 | tail -1 | sed "s/^/PushT2dHard lm$lmv run$run: /"
  done
done
for lmv in 1 2; do $H/scripts/lplf.sh AllegroPen 0:1 E3lm${lmv}p e3.jsonl $V --lm $lmv --log-steps 1 | tail -1 | sed "s/^/pen lm$lmv: /"; done
# reference: deterministic ops, current lfddp_v.py (defaults) vs the committed V14 file
for sc in lfddp_v.py lfddp_v_44ca.py; do
  DET=1 SCRIPT=$sc $H/scripts/vplf.sh PushT2dHard 0:3 REF_${sc%.py} ref.jsonl $V > /dev/null
  DET=1 SCRIPT=$sc $H/scripts/vplf.sh PushSquareKnobTable 10:11 REF_${sc%.py} ref.jsonl $V > /dev/null
done
echo E3DONE
