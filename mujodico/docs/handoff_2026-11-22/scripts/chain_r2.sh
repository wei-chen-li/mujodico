#!/bin/bash
# re-solve option 2: LPEC rounds start at the fine level (--lpec-start fine) vs the coarse excursion (V14), from stuck plans
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
P=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300 SCRIPT=lfddp_v_r2.py
for TS in PushT2dHard:23:GCn PushT2dHard:25:GCn PushT2d:10:GCn PushT2d:11:GCn PushT2dHardLong:10:GCn PushT2dHardLong:11:GCn PushT2dHard:4:V14; do
  IFS=: read t s src <<< "$TS"
  for mode in coarse fine; do
    $H/scripts/vplf.sh $t $s:$((s+1)) R2${mode} r2.jsonl $V --from-us $P/${src}_${t}_{seed}.npz --lpec-start $mode | head -1 | cut -c1-150
  done
done
for src in GN4 PV11d PV11c; do for mode in coarse fine; do
  $H/scripts/vplf.sh AllegroPen 0:1 R2${mode}_$src r2.jsonl $V --from-us $P/${src}_AllegroPen_{seed}.npz --lpec-start $mode | head -1 | cut -c1-150
done; done
echo R2DONE
