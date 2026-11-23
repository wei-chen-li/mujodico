#!/bin/bash
# (3) throughput: 12 PushT2dHard seeds, one GPU process vs 3 concurrent GPU processes vs 4 CPU-only processes
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/vcycle; V="$(cat $R/V13flags.txt) --ip-hess-cycle 0"
while pgrep -f "chain_f[.]sh" >/dev/null; do sleep 5; done
t0=$(date +%s.%N); $H/scripts/vplf.sh PushT2dHard 6:18 PA1 par.jsonl $V | tail -1; t1=$(date +%s.%N)
echo "A one GPU process: $(echo "$t1 - $t0" | bc) s"
t0=$(date +%s.%N)
for r in 6:10 10:14 14:18; do $H/scripts/vplf.sh PushT2dHard $r PB_${r/:/_} par.jsonl $V > /dev/null & done; wait; t1=$(date +%s.%N)
echo "B 3 GPU processes: $(echo "$t1 - $t0" | bc) s"
JAX_PLATFORMS=cpu $H/scripts/vplf.sh PushT2dHard 6:7 PCwarm par.jsonl $V | tail -1   # warm the CPU export/compile cache
t0=$(date +%s.%N)
for r in 6:9 9:12 12:15 15:18; do JAX_PLATFORMS=cpu $H/scripts/vplf.sh PushT2dHard $r PC_${r/:/_} par.jsonl $V > /dev/null & done; wait; t1=$(date +%s.%N)
echo "C 4 CPU processes: $(echo "$t1 - $t0" | bc) s"
echo GDONE
