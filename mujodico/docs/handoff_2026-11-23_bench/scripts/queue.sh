#!/bin/bash
# usage: queue.sh <jobfile>   each line: <Task> <a:b> <label> <out.jsonl> [extra flags]
# Runs jobs one at a time through run_plfv.sh; before each job waits while the other session's
# wall-time windows is open (chain_lad.txt: E4DONE..E6DONE, E7DONE..E5DONE); a running job is stopped (whole process group) when a window opens.
LAD=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/results/ladder/chain_lad.txt
H=$(dirname "$(readlink -f "$0")")
in_window() {   # other session's wall-time windows: E4DONE..E6DONE (E6) and E7DONE..E5DONE (E5)
  { grep -q E4DONE "$LAD" && ! grep -q E6DONE "$LAD"; } || { grep -q E7DONE "$LAD" && ! grep -q E5DONE "$LAD"; }
}
gate() {
  while [ -s "$LAD" ] && in_window; do echo "[queue] $(date +%T) waiting: chain_lad timing window open"; sleep 60; done
}
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  gate
  echo "[queue] $(date +%T) start: $line"
  setsid $H/run_plfv.sh $line &      # own process group, so the watchdog can stop the whole job
  JP=$!
  while kill -0 $JP 2>/dev/null; do
    if [ -s "$LAD" ] && in_window; then
      echo "[queue] $(date +%T) window opened: stopping job (pgid $JP): $line"; kill -- -$JP; wait $JP; break
    fi
    sleep 5
  done
  wait $JP 2>/dev/null
  echo "[queue] $(date +%T) done: $line"
done < "$1"
echo "[queue] ALLDONE"
