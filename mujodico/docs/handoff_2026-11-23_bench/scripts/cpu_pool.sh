#!/bin/bash
# usage: cpu_pool.sh <jobfile> <log> [K=4] [first_core=12]
# Each line of <jobfile> is a command (run from the repo root mujodico-main-dev). Runs up to K at once, each pinned
# to 2 CPU cores starting at first_core. Logs "start/done <line>" lines.
JF=$1; LOG=$2; K=${3:-4}; C0=${4:-12}
cd /home/gchou/PycharmProjects/mujodico-main-dev
declare -A slot
i=0
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  while true; do
    for s in $(seq 0 $((K-1))); do
      p=${slot[$s]}
      if [ -z "$p" ] || ! kill -0 $p 2>/dev/null; then
        c=$((C0 + 2*s)); echo "[pool] $(date +%T) start (cores $c,$((c+1))): $line" >> $LOG
        ( taskset -c $c,$((c+1)) bash -c "$line" > /dev/null 2>&1; echo "[pool] $(date +%T) done: $line" >> $LOG ) &
        slot[$s]=$!; break 2
      fi
    done
    sleep 5
  done
done < "$JF"
wait
echo "[pool] ALLDONE" >> $LOG
