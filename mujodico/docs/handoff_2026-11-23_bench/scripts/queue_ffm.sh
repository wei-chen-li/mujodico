#!/bin/bash
# usage: queue_ffm.sh <jobfile>: like queue.sh, but each job's far-field margin comes from its label suffix
# (V15LF03 -> FFM=0.03, V15LF05 -> 0.05, V15LF10 -> 0.10); FULLLAW=1 (all constraint forces lifted).
H=$(dirname "$(readlink -f "$0")")
while read -r T R L O; do
  [ -z "$T" ] && continue
  f=${L##*LF}; FFM="0.${f}"
  echo "[ffm] $(date +%T) start: $T $R $L (FFM=$FFM)"
  FULLLAW=1 FFM=$FFM MEMFRAC=0.15 $H/run_plfv.sh $T $R $L $O
  echo "[ffm] $(date +%T) done: $T $R $L"
done < "$1"
echo "[ffm] ALLDONE"
