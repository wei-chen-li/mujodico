#!/bin/bash
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V=$(cat $R/V13flags.txt)
cd /home/gchou/PycharmProjects/mujodico-main-dev
while pgrep -f chain_gn.sh >/dev/null; do sleep 5; done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# (a) per-iteration IP cost, exact vs Gauss-Newton dynamics Hessian
for t in AllegroPen PushT2dHard AllegroJarC; do for h in 1 0; do .venv/bin/python $H/scripts/ip_bench.py $t $h 60 2>&1 | grep "ms/it"; done; done > $R/logs/ip_bench.txt
# (b) LPEC folded in: from the stuck slide PV11d (driver end to end: 0.0062)
GRB_TL=300 $H/scripts/vplf.sh AllegroPen 0:1 FLPEC_PV11d speed.jsonl $V --from-us $R/plans/PV11d_AllegroPen_{seed}.npz --lpec 4
# (c) read-out overlap exactness under deterministic ops
DET=1 $H/scripts/vplf.sh PushT2dHard 0:3 OV0det speed.jsonl $V
DET=1 $H/scripts/vplf.sh PushT2dHard 0:3 OV1det speed.jsonl $V --ro-overlap 1
