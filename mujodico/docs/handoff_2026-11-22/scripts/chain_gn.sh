#!/bin/bash
# GN vs exact IP Hessian, sequential (clean timing)
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
V=$(cat $H/results/vcycle/V13flags.txt)
while pgrep -f "lfddp_v.py AllegroPen 0:1 EX1" >/dev/null; do sleep 5; done
$H/scripts/vplf.sh AllegroPen 0:1 GN2 speed.jsonl $V --ip-hess 0
$H/scripts/vplf.sh AllegroPen 0:1 EX2 speed.jsonl $V
$H/scripts/vplf.sh AllegroPen 0:1 GN3 speed.jsonl $V --ip-hess 0
$H/scripts/vplf.sh PushT2dHard 0:6 GN1 speed.jsonl $V --ip-hess 0
$H/scripts/vplf.sh PushT2dHard 0:6 EX1 speed.jsonl $V
$H/scripts/vplf.sh PushT2d 0:6 GN1 speed.jsonl $V --ip-hess 0
