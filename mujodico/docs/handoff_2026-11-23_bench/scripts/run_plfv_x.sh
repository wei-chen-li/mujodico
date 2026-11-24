#!/bin/bash
# run_plfv_x.sh: as run_plfv.sh, but the BASE flags can be replaced via env XBASE (lfddp_v takes the FIRST occurrence of a flag)
# usage: run_plfv.sh <Task> <a:b> <label> <out.jsonl-name> [extra lfddp_v flags]
# V15 = vplf.sh BASE + default task class + V15flags; rows -> results/<out>, logs -> results/logs, plans -> results/plans
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-23_bench
cd /home/gchou/PycharmProjects/mujodico-main-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.25}   # hard cap (~4 GB): fail fast rather than crowd the other session
unset XLA_FLAGS
mkdir -p $H/results/logs $H/results/plans
T=$1; R=$2; L=$3; O=$4; shift 4
BASE="${XBASE:---warm pip --warm-ip 80 --lm 1 --mistrust 1 --dwell 10 --par-ls 1 --ro-every 2 --cycle 2}"
CLS="--mu0 1e-3 --handoff-from incumbent"
V15="$(cat $H/../handoff_2026-11-22/results/vcycle/V15flags.txt)"
FULLLAW=$FULLLAW FFM=$FFM SAVE_DIR=$H/results/plans .venv/bin/python $H/scripts/plfv.py $T $R $L $H/results/$O $BASE $CLS $V15 "$@" > $H/results/logs/log_${L}_${T}.txt 2>&1
grep "^{" $H/results/logs/log_${L}_${T}.txt | python3 -c "
import sys, json
n=0; ok=0
for l in sys.stdin:
    r = json.loads(l); n+=1; ok+=r['true_ok']
    print(r['label'], r['task'], r['seed'], 'OK  ' if r['true_ok'] else 'miss', round(r['true_goal'],4), r['reason'], 'cyc', r.get('cycles'), 'its', r['it']+r['ip_its'], 't', round(r['time'],1))
print('solved', ok, '/', n)
"
