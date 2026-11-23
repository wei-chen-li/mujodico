#!/bin/bash
# validation: V14 + structured-QP LPEC (--lpec-solver fast) + parallel trust radii (--lpec-par 3), from the stuck plans of chain_r2
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
P=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
while pgrep -f "chain_r2[.]sh" >/dev/null; do sleep 10; done
for TS in PushT2dHard:23:GCn PushT2dHard:25:GCn PushT2d:10:GCn PushT2dHardLong:10:GCn PushT2dHardLong:11:GCn PushT2dHard:4:V14; do
  IFS=: read t s src <<< "$TS"
  $H/scripts/vplf.sh $t $s:$((s+1)) LVfast r2.jsonl $V --from-us $P/${src}_${t}_{seed}.npz --lpec-solver fast --lpec-par 3 | head -1 | cut -c1-150
done
for src in GN4 PV11d PV11c; do
  $H/scripts/vplf.sh AllegroPen 0:1 LVfast_$src r2.jsonl $V --from-us $P/${src}_AllegroPen_{seed}.npz --lpec-solver fast --lpec-par 3 | head -1 | cut -c1-150
done
echo LVDONE
