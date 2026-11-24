#!/bin/bash
# E6 step 2 (HANDOFF_ladder_search.md §5 E6): the T instances of the E6 set with a cap on the cycles of each LPEC round.
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/ladder; PV=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V15flags.txt)
export GRB_TL=300 SCRIPT=lfddp_l_e6.py
L=$H/scripts/lplf.sh
for c in 2 3; do
  for TS in PushT2dHard:23 PushT2dHard:25 PushT2d:10 PushT2d:11 PushT2dHardLong:10 PushT2dHardLong:11; do
    IFS=: read t s <<< "$TS"
    $L $t $s:$((s+1)) E6c$c e6cap.jsonl $V --lpec-cyc $c --from-us $PV/GCn_${t}_{seed}.npz | head -1 | cut -c1-100 >> $R/chain_e6cap.txt
  done
  $L PushT2dHard 0:1 E6c$c e6cap.jsonl $V --lpec-cyc $c | head -1 | cut -c1-100 >> $R/chain_e6cap.txt
  $L PushT2dHard 4:5 E6c$c e6cap.jsonl $V --lpec-cyc $c | head -1 | cut -c1-100 >> $R/chain_e6cap.txt
done
echo E6CAPDONE >> $R/chain_e6cap.txt
