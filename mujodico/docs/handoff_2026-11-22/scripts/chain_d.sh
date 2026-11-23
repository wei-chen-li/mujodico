#!/bin/bash
# GN in fidelity-cycle excursions only (--ip-hess-cycle 0): base path = PLF where no cycle fires
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V13flags.txt)
while pgrep -f "chain_c[.]sh" >/dev/null; do sleep 5; done
for L in GC1 GC2 GC3; do $H/scripts/vplf.sh PushT2dHard 0:6 $L speed.jsonl $V --ip-hess-cycle 0 | tail -1; done
$H/scripts/vplf.sh PushT2dHardLong 0:6 GC1 speed.jsonl $V --ip-hess-cycle 0 | tail -1
$H/scripts/vplf.sh PushT2d 0:6 GC1 speed.jsonl $V --ip-hess-cycle 0 | tail -1
for L in GC1 GC2 GC3; do $H/scripts/vplf.sh AllegroPen 0:1 $L speed.jsonl $V --ip-hess-cycle 0 | tail -1; done
$H/scripts/vplf.sh PushBox2d 0:6 GN2 speed.jsonl $V --ip-hess 0 | tail -1
echo DDONE
