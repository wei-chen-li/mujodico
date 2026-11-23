#!/bin/bash
# lazy parallel radii (first radius alone, the rest concurrently after a miss), Gurobi vs structured QP, from the stuck plans
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
P=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V14flags.txt)
export GRB_TL=300
while pgrep -f "chain_e3[.]sh" >/dev/null; do sleep 10; done
for solver in grb fast; do
  for TS in PushT2dHard:23:GCn PushT2dHard:25:GCn PushT2d:10:GCn PushT2dHardLong:10:GCn PushT2dHardLong:11:GCn PushT2dHard:4:V14; do
    IFS=: read t s src <<< "$TS"
    $H/scripts/vplf.sh $t $s:$((s+1)) LZ$solver r2.jsonl $V --from-us $P/${src}_${t}_{seed}.npz --lpec-solver $solver --lpec-par 3 --lpec-par-lazy 1 | head -1 | cut -c1-120
  done
  $H/scripts/vplf.sh AllegroPen 0:1 LZ${solver}_GN4 r2.jsonl $V --from-us $P/GN4_AllegroPen_{seed}.npz --lpec-solver $solver --lpec-par 3 --lpec-par-lazy 1 | head -1 | cut -c1-120
done
echo LZDONE
