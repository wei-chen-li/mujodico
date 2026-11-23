#!/bin/bash
# Gauss-Newton IP Hessian (--ip-hess 0): validation on the gate, T family, Allegro jar and pen (sequential)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V13flags.txt)
while pgrep -f "chain_b[.]sh" >/dev/null; do sleep 5; done
$H/scripts/vgate.sh GN1 gate_gn.jsonl $V --ip-hess 0
for L in GN2 GN3; do $H/scripts/vplf.sh PushT2dHard 0:6 $L speed.jsonl $V --ip-hess 0 | tail -1; done
$H/scripts/vplf.sh PushT2dHardLong 0:6 GN1 speed.jsonl $V --ip-hess 0 | tail -1
$H/scripts/vplf.sh AllegroJar 0:5 GN1 speed.jsonl $V --ip-hess 0 | tail -1
$H/scripts/vplf.sh AllegroJarC 0:5 GN1 speed.jsonl $V --ip-hess 0 --clear-margin 0.003 | tail -1
for L in GN4 GN5 GN6; do $H/scripts/vplf.sh AllegroPen 0:1 $L speed.jsonl $V --ip-hess 0 | tail -1; done
echo CDONE
