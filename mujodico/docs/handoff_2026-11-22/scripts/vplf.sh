#!/bin/bash
# usage: vplf.sh <Task> <a:b> <label> <out.jsonl-name> [extra lfddp_v.py flags]
# PLF settings of HANDOFF_trust_ratio.md §14.5 by task class + extra flags; rows -> results/vcycle/<out>, plans -> results/vcycle/plans
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
cd /home/gchou/PycharmProjects/mujodico-main-dev
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -z "$DET" ] && unset XLA_FLAGS || export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false"
# CPU1T=1: single-threaded Eigen on the CPU devices (read-outs, line search, backward pass); roundoff-level changes on Allegro
[ -n "$CPU1T" ] && export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=false"
T=$1; R=$2; L=$3; O=$4; shift 4
BASE="--warm pip --warm-ip 80 --lm 1 --mistrust 1 --dwell 10 --par-ls 1 --ro-every 2 --cycle 2"
case $T in
  PushT2d*|PushBox2d) CLS="--mu0 1e-2 --ip-mu 0.1 --ip-mumin 0.1 --handoff-from true --restart true --lin-dev cpu";;
  *Knob*) CLS="--sig0 10 --mu0 1e-2 --ip-mu 0.1 --ip-mumin 0.1 --lin-dev cpu --handoff-from incumbent";;
  PushSphere*|PushMulti*) CLS="--mu0 1e-2 --lin-dev cpu --handoff-from incumbent";;
  *) CLS="--mu0 1e-3 --handoff-from incumbent";;
esac
SAVE_DIR=$H/results/vcycle/plans .venv/bin/python $H/scripts/${SCRIPT:-lfddp_v.py} $T $R $L $H/results/vcycle/$O $BASE $CLS "$@" > $H/results/vcycle/logs/log_${L}_${T}${LOGSUF}.txt 2>&1
grep "^{" $H/results/vcycle/logs/log_${L}_${T}${LOGSUF}.txt | python3 -c "
import sys, json
n=0; ok=0
for l in sys.stdin:
    r = json.loads(l); n+=1; ok+=r['true_ok']
    print(r['label'], r['task'], r['seed'], 'OK  ' if r['true_ok'] else 'miss', round(r['true_goal'],4), r['reason'], 'cyc', r.get('cycles'), ' '.join(f\"[s{c['sig']:g} {c['its']}{c['hand'][0]}: {c['start']:.3f}->{c['after_ip']:.3f}]\" for c in r.get('cyc_log',[])), 'its', r['it']+r['ip_its'], 't', round(r['time'],1))
print('solved', ok, '/', n)
"
