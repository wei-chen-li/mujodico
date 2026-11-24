#!/bin/bash
# E8 (HANDOFF_ladder_search.md §4.8, §5 E8): IP vs lifted (carried-gap) excursion engine from cycle-start incumbents.
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/ladder; PV=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V15flags.txt)
export SCRIPT=lfddp_l_e8.py
L=$H/scripts/lplf.sh
# 1. cycle-start incumbents (saved to results/ladder/plans/E8g_<Task>_<seed>_c<n>.npz)
$L PushT2dHard 0:30 E8g e8gen.jsonl $V --lpec 0 --save-cyc 1 | tail -1 | sed "s/^/E8g /" >> $R/chain_e8.txt
# 2. one sigma0 excursion from each incumbent, IP and lifted (fixed nu, and costate nu)
one() {   # task seed file tag
  for eng in ip lifted lifted_lam; do
    ex="--exc-only ${eng%%_*}"; [ $eng = lifted_lam ] && ex="$ex --exc-nu lam"
    LOGSUF=_$4_$eng $L $1 $2:$(($2+1)) E8x_$eng e8.jsonl $V --lpec 0 $ex --from-us $3 > /dev/null
  done
}
for f in $R/plans/E8g_PushT2dHard_*_c*.npz; do b=$(basename $f .npz); s=$(echo $b | cut -d_ -f3); one PushT2dHard $s $f $b; done
for ts in PushT2dHard:23 PushT2dHard:25 PushT2dHardLong:10 PushT2dHardLong:11 PushT2d:10 PushT2d:11; do
  t=${ts%%:*}; s=${ts##*:}; one $t $s $PV/GCn_${t}_$s.npz GCn_${t}_$s; done
echo E8TDONE >> $R/chain_e8.txt
for src in PV11c PV11d PV2; do one AllegroPen 0 $PV/${src}_AllegroPen_0.npz pen_$src; done
echo E8DONE >> $R/chain_e8.txt
