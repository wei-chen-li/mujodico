#!/bin/bash
# HANDOFF_ladder_search.md: E2, E4, E6, E7, E5 with the frozen driver lfddp_l_f.py (V15 flags); after E1
H=/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22
R=$H/results/ladder; PV=$H/results/vcycle/plans; V=$(cat $H/results/vcycle/V15flags.txt)
export GRB_TL=300 SCRIPT=lfddp_l_f.py
L=$H/scripts/lplf.sh
while pgrep -f "chain_e1[.]sh" >/dev/null; do sleep 15; done
arm() {   # label rowsfile extra-flags...
  lab=$1; out=$2; shift 2
  $L FingersSphere2dRandomizedHardLong 0:24 $lab $out $V "$@" | tail -1 | sed "s/^/$lab fingers-hard: /"
  $L PushBox2d 0:6 $lab $out $V "$@" | tail -1 | sed "s/^/$lab PushBox2d: /"
  $L PushT2dHard 0:12 $lab $out $V "$@" | tail -1 | sed "s/^/$lab PushT2dHard: /"
  for p in 1 2 3; do $L AllegroPen 0:1 ${lab}p$p $out $V "$@" | tail -1 | sed "s/^/$lab pen$p: /"; done
}
# ---- E2 ----
arm E2base e2.jsonl
arm E2a e2.jsonl --prog-stall 2
arm E2b e2.jsonl --watchdog-all 1
arm E2c e2.jsonl --cycle 0 --fine-rung 1
arm E2abc e2.jsonl --prog-stall 2 --watchdog-all 1 --cycle 0 --fine-rung 1
echo E2DONE
# ---- E4 (knob base too) ----
$L PushSquareKnobTable 0:20 E2base e4.jsonl $V | tail -1 | sed "s/^/E4base knob: /"
PLF_FN_EXPORT=0 arm E4mult e4.jsonl --term mult
PLF_FN_EXPORT=0 $L PushSquareKnobTable 0:20 E4mult e4.jsonl $V --term mult | tail -1 | sed "s/^/E4mult knob: /"
arm E4b0 e4.jsonl --boosts 0
$L PushSquareKnobTable 0:20 E4b0 e4.jsonl $V --boosts 0 | tail -1 | sed "s/^/E4b0 knob: /"
echo E4DONE
# ---- E6: logged rescues from the stuck-plan set (V15) ----
for TS in PushT2dHard:23 PushT2dHard:25 PushT2d:10 PushT2d:11 PushT2dHardLong:10 PushT2dHardLong:11; do
  IFS=: read t s <<< "$TS"
  $L $t $s:$((s+1)) E6 e6.jsonl $V --from-us $PV/GCn_${t}_{seed}.npz | head -1 | cut -c1-100
done
$L PushT2dHard 0:1 E6 e6.jsonl $V | head -1 | cut -c1-100
$L PushT2dHard 4:5 E6 e6.jsonl $V | head -1 | cut -c1-100
for src in PV11c PV11d PV2; do $L AllegroPen 0:1 E6_$src e6.jsonl $V --from-us $PV/${src}_AllegroPen_{seed}.npz | head -1 | cut -c1-100; done
echo E6DONE
# ---- E7: stop classes and fixed points (deterministic ops) ----
DET=1 $L PushT2dHard 0:30 E7 e7.jsonl $V | tail -1 | sed "s/^/E7 PushT2dHard: /"
DET=1 $L PushT2dHardLong 0:18 E7 e7.jsonl $V | tail -1 | sed "s/^/E7 Long: /"
/home/gchou/PycharmProjects/mujodico-main-dev/.venv/bin/python - <<'PY' > $R/e7_restart_list.txt
import json
for l in open('/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/results/ladder/e7.jsonl'):
  r = json.loads(l)
  if r['label'] == 'E7' and r['stop_class'] != 'goal':
    print(r['task'], r['seed'], r['stop_class'])
PY
while read t s cls; do
  DET=1 $L $t $s:$((s+1)) E7fp e7.jsonl $V --from-us $R/plans/E7_${t}_{seed}.npz | head -1 | cut -c1-100 | sed "s/^/[$cls] /"
done < $R/e7_restart_list.txt
for s in 1 2 3 5 7; do DET=1 $L PushT2dHard $s:$((s+1)) E7fp e7.jsonl $V --from-us $R/plans/E7_PushT2dHard_{seed}.npz | head -1 | cut -c1-100 | sed "s/^/[goal] /"; done
echo E7DONE
# ---- E5: discovered top rung (two runs) ----
for run in 1 2; do
  for TR in PushSphere2d:0:10 PushSphere2dHard:0:10 PushMultiSphere2d:0:6 PushBox2d:0:6 PushSquareKnobTable:0:20 FingersSphere2dRandomized:0:24 FingersSphere2dRandomizedHardLong:0:24 AllegroBallTableHard:0:10 PushT2dHard:0:12; do
    IFS=: read t a b <<< "$TR"
    $L $t $a:$b E5r$run e5.jsonl $V --ladder auto | tail -1 | sed "s/^/E5 run$run $t: /"
  done
done
echo E5DONE
