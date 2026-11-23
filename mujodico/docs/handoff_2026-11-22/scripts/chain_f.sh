#!/bin/bash
# candidate final: V13 + GN in cycle excursions; gate, LPEC fallback from a stuck slide, pen from scratch with --lpec
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V13flags.txt)
while pgrep -f "chain_e[.]sh" >/dev/null; do sleep 5; done
$H/scripts/vgate.sh GC1 gate_gc.jsonl $V --ip-hess-cycle 0
GRB_TL=300 $H/scripts/vplf.sh AllegroPen 0:1 FLGC_EX2 speed.jsonl $V --ip-hess-cycle 0 --from-us $R/plans/EX2_AllegroPen_{seed}.npz --lpec 6 | tail -1
GRB_TL=300 $H/scripts/vplf.sh AllegroPen 0:1 FLGC_GN4 speed.jsonl $V --ip-hess-cycle 0 --from-us $R/plans/GN4_AllegroPen_{seed}.npz --lpec 6 | tail -1
echo FDONE
