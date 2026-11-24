#!/bin/bash
# E8 / F8: the lifted engine with the IP's globalization (filter) and with the fraction-to-boundary step (F8.1 + F8.2),
# on the inputs of chain_e8.sh; starts after chain_e8 prints E8DONE.
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/ladder; PV=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V15flags.txt)
export SCRIPT=lfddp_l_e8b.py
L=$H/scripts/lplf.sh
until grep -q E8DONE $R/chain_e8.txt 2>/dev/null; do sleep 60; done
one() {
  for eng in lifted_filter lifted_ff; do
    ex="--exc-only lifted --exc-glob filter"; [ $eng = lifted_ff ] && ex="$ex --exc-ftb 1"
    LOGSUF=_$4_$eng $L $1 $2:$(($2+1)) E8x_$eng e8.jsonl $V --lpec 0 $ex --from-us $3 > /dev/null
  done
}
for f in $R/plans/E8g_PushT2dHard_*_c*.npz; do b=$(basename $f .npz); s=$(echo $b | cut -d_ -f3); one PushT2dHard $s $f $b; done
for ts in PushT2dHard:23 PushT2dHard:25 PushT2dHardLong:10 PushT2dHardLong:11 PushT2d:10 PushT2d:11; do
  t=${ts%%:*}; s=${ts##*:}; one $t $s $PV/GCn_${t}_$s.npz GCn_${t}_$s; done
for src in PV11c PV11d PV2; do one AllegroPen 0 $PV/${src}_AllegroPen_0.npz pen_$src; done
echo E8BDONE >> $R/chain_e8.txt
