# Handoff: fidelity cycles (re-opening discovery from the incumbent) — implementation and test plan

Date: 2026-11-22 (twelfth session of the day). Repo state: `master` at `f2fdc58` plus the uncommitted opt-ins of `../handoff_2026-11-14/followup/PRINCIPLES.md`. No package source was modified by the pilot described here.

Read first: `HANDOFF_trust_ratio.md` §14–§18 in this folder (PLF, PIP4F, F5, the hard tasks, the LPEC and per-face experiments). This document starts where §18 stops.

**Status (updated 2026-11-23, day session; results in §11).** Recommended: PLF-V with Gauss–Newton dynamics curvature in the fidelity-cycle excursions and the LPEC fallback inside the solver (`scripts/lfddp_v.py`, flags `results/vcycle/V14flags.txt`): PushT2dHard 29/30 (seeds 0–29), PushT2d 18/18, PushT2dHardLong 17/18 (seeds 0–17), AllegroPen 5/5 from scratch (PLF-V alone 11/12 with this setting, 70–131 s), gate 90/90 and knob 20/20 with PLF's path unchanged wherever no cycle fires. Night-session status (§10): PLF-V, the fidelity cycle inside PLF (§4.2 with the changes of §10.2), is implemented in `scripts/lfddp_v.py`. With one flag set for all tasks: PushT2d 24/24, PushT2dHard 37/42, PushT2dHardLong 16/18 (pooled over 3–7 runs), AllegroPen solved from scratch by PLF-V alone in 14 of 19 runs, and 5 of the 6 stuck slides by an LPEC branch step + PLF-V (§10.11; the sixth reached 0.040); gate 90/90 and knob 20/20 in four runs, and identical to PLF wherever no cycle fires. Recommended limit 10 cycles (§10.12: PushT2dHard 23/24, Long 6/6, PushT2d 6/6, pen 3/4 by PLF-V alone, gate 90/90 + knob 20/20). The pilot of §1 is kept below unchanged.

## 0. What is being asked, and the short version

**Question.** Is there a reformulation of the solver, the contact formulation or the optimization formulation that solves PushT2d, PushT2dHard and AllegroPen, under these constraints from the user:

- no random sampling, no heuristic initializations, no task-specific method changes;
- local optimization, with local branching or combinatorics if needed;
- no enumeration of contact sequences in the first pass (batched GPU enumeration may be considered later);
- one unified story;
- no substantial increase in solve time.

**Answer so far.** In every current recipe (F5, PIP4F, PLF) contact discovery happens once per solve. After the first accepted improvement the smoothing only goes down and the contact-attraction barrier is dropped. A multi-stroke task needs discovery once per stroke. Restarting the coarse level from the incumbent, with true-law acceptance, solved the pen (0.597 → 0.0082) and raised PushT2dHard from 1–3/6 to 3–4/6 in single runs. It is not yet a full solution and it currently wastes most of its iterations.

**What this plan does.**

1. Checks that the pilot gain is a mechanism and not re-roll noise (E0, E1).
2. Finds which ingredient of the coarse restart does the work: smoothing, barrier, or both (E2).
3. Makes the restart cheap: a cap and a stall rule for unproductive excursions, a computed test that picks the level without a trial, and LFDDP for the convergence after an escape (E3–E6).
4. Makes it local in time: coarse only on the stretches where the incumbent has no contact, so established strokes are not disturbed (E7, E8).
5. Tests whether the same rule removes the per-task barrier and starting-smoothing settings (E9).
6. Holds every step to a solve-rate gate and a speed gate (§4.7, E10).
7. Gives a fallback for each step that can fail (§6).

**Terms used below.**

| Term | Meaning |
|---|---|
| Incumbent | Best controls so far and their open-loop true-law rollout. Always an exact true-law trajectory. |
| Level | The smoothing multiplier of the planning law (1 = the task's planning smoothing, 0 = true law). "Coarse" = 1 or 10. "Fine" = 0.001. |
| Discovery barrier | The barrier setting used before the first accepted improvement: `mu_init = mu0`, `mu_min = 1e-2`. The post-acceptance setting is `mu_init = 1e-3`, `mu_min = 1e-4`. |
| Cycle | One restart of the ladder from the incumbent at a coarse level with the discovery barrier, followed by the usual sharpening. |
| Excursion | The solver run at the start of a cycle, before its first accepted improvement. |
| Productive cycle | A cycle with at least one accepted improvement of the true goal. |
| Blind | The current model predicts no decrease at the incumbent although the goal is far. |
| Idle stretch | Consecutive stages of the incumbent's true rollout with no contact force. |
| Gate set | The 90 seeds of `HANDOFF_trust_ratio.md` §13 (fingers-hard, fingers, PushSphere2d, PushSphere2dHard, AllegroBallTableHard, PushMultiSphere2d, PushBox2d). |

## 1. Evidence this plan is built on

All runs: one job on the GPU unless noted, default XLA flags, single runs. Rows are in `results/vcycle/vc.jsonl` (automatic rule) and `results/vcycle/cyc.jsonl` (hand-driven chain). Plans are in `results/vcycle/plans/`, logs in `results/vcycle/logs/`.

### 1.1 Failing T plans end in idle tails

`scripts/strokes.py` on the saved plans of `results/hard/plans/` (contact strokes along the true rollout):

| Plan | Goal | Strokes (steps) | Idle steps at the end |
|---|---|---|---|
| PIP4F PushT2dHard seed 3 | 0.713 | bar−x [7], bar−y [10–17] | 123 of 140 |
| PIP4F PushT2dHard seed 4 | 2.956 | bar−x [22–27] | 113 |
| PLF PushT2dHard seed 3 | 0.280 | bar−y [14–34] | 106 |
| F5 PushT2dHard seed 0 | 0.794 | stem−x, four short pushes [29–53] | 87 |
| PLF PushT2d seed 0 | 0.399 | bar+x [26–32] | 68 of 100 |
| F5 PushT2d seeds 0, 1, 3 | 0.30–0.75 | none (the pusher never reaches the T) | 100 |

Not every miss has an idle tail. PIP4F PushT2dHard seed 0 (0.540) pushes until step 138.

### 1.2 Discovery is one-shot in the code

- `scripts/c1_accept.py`, the `solve_ph(...)` call inside the burst loop: after any accepted burst the phase is `PH(sm, mu_burst, 1e-4)`. Before it, the phase is `PH(sm, mu0, disc_mumin)`. Accepted bursts move one level down. Rejects move up one level but halve `k`, so a coarse level is only revisited with short bursts and without the discovery barrier.
- `scripts/lfddp.py`, the `cycle` branch of the model-change block: `explore(u_inc, Xr, Cr, sig, boosts, ip_cycle)` re-enters the lifted IP at the **current** `sig` (fine by then) and the current terminal-weight boost.
- `scripts/pip.py`, `--restart-exh`: a fresh IP solve from the incumbent "at the current model". `--widen` raises barrier and smoothing only while the incumbent is within 25 % of the start.

Earlier observations that fit: restarts from a failing incumbent reproduce its stroke pattern (§16.1 of `HANDOFF_trust_ratio.md`, PIP4F seed 3); PLF's cycle and the no-branch LPEC step return to the pen slide (§18.1).

### 1.3 Pilot: cycles on top of F5

Script `scripts/c1_vcycle.py` (= `c1_accept.py` with a cycle loop; env `NCYC` = maximum cycles). F5's flags, `--up 2` so that level 10 exists. Rule at each stop without success:

- goal < 0.03: restart at level 0.1 with the post-acceptance barrier ("near");
- else if the last cycle improved the goal by ≥ 10 %: restart at level 1 with the discovery barrier;
- else if level 10 has not been tried since the last improvement: restart at level 10;
- else stop.

Each cycle has F5's own budget of 250 iterations. Acceptance is unchanged (true goal must improve), so the result is never worse than the starting plan.

| Case | Before | With cycles (`NCYC=6`) |
|---|---|---|
| AllegroPen, from the PLF slide plan | 0.597, pen at z 2.5 cm | 0.197 after one level-1 cycle; continued at the fine level: **0.0082, pen at z 9.3 cm** (goal z 10 cm) |
| PushT2dHard, from PIP4F plans | 1/6 | 4/6 (misses 0.061, 0.130) |
| PushT2dHard, from PLF plans | 2/6 | 3/6 (misses 0.048, 0.132, 0.825) |
| PushT2dHard, F5 from scratch | 3/6 at cycle 0 | 4/6 (misses 0.052, 0.240) |
| PushT2dHardLong, F5 from scratch | 1/6 at cycle 0 | 3/6 (misses 0.0101, 0.0157, 0.133) |
| PushT2d, from PLF plans | 5/6 | 6/6 (seed 0: 0.399 → 0.0058 in one cycle) |
| PushT2d, from PIP4F plans | 5/6 | 5/6 (seed 3 stays at 0.1855) |
| PushT2d, F5 from scratch | 3/6 at cycle 0 | 3/6 (seed 0: 0.734 → 0.069; seeds 1, 3 never move) |

- Progress accumulates over cycles. Example, F5 from scratch, PushT2dHard seed 4: 3.60 → 3.10 → 2.00 → 1.38 → 0.257 → 0.052.
- Cycled plans relocate to far faces. Hand-driven chain from PIP4F PushT2dHard seed 4: bar−x [22–27] became bar−x [21–22], bar−y [26–84], bar+y [98–140], goal 0.0020.
- Pen detail (`logs/log_VPEN_AllegroPen.txt`, `log_VPEN2_AllegroPen.txt`, check with `scripts/chk_pen.py`):
  - level-1 excursion: checkpoints at 11 and 32 iterations rejected, accepted at 73 (0.514);
  - then every burst at 0.1, 0.01 and 0.001 accepted, down to 0.197, when the 250-iteration budget ended the cycle;
  - the automatic rule then restarted at level 1 and at level 10 and wasted about 500 iterations (true goals between 0.52 and 3·10⁴);
  - a separate fine-level run from the 0.197 plan (`--ladder 0.001 --budget 600`, post-acceptance barrier) reached 0.0082 in 524 iterations, 2–5 % per accepted burst.
  - The success metric weights pen position 100 and orientation 0.01, so 0.0082 certifies the lift, not the upright orientation.

### 1.4 Where the iterations go

From the logs of the automatic-rule runs (71 restart cycles; a from-scratch cycle 0 is not counted, a first cycle from a saved plan is):

| Restart level | Cycles | Productive | Iterations to the first accept in productive cycles |
|---|---|---|---|
| 1 | 45 | 23 | ≤ 32 in 11, ≤ 73 in 15, ≤ 154 in 19, 255 in 4 |
| 10 | 22 | 9 | ≤ 32 in 3, ≤ 73 in 6, ≤ 154 in 9 |
| 0.1 (near) | 4 | 3 | 11, 54, 54 |

- 8,559 of 13,863 restart iterations (62 %) were spent in unproductive cycles. An unproductive coarse cycle costs about 255 iterations (the burst caps 10, 20, 40, 80, 100).
- Accept checks happen only at burst ends (cumulative 11, 32, 73, 154, 255 iterations), so "first accept" is an upper bound on when the true goal first improved.
- Eight bursts produced true rollouts with goal > 10, all on the pen: seven in the two coarse restarts from the 0.197 plan, one at the fine level. Acceptance rejects them, but they are paid for. No T run produced one.

### 1.5 Timing

- T tasks: 13–25 ms per solver iteration; true rollouts, retractions and TVLQR together < 10 % of the time. The first seed of a process pays 10–90 s of cache loading or compiling.
- Seeds solved in cycle 0 pay nothing: the cycle loop is entered only after the base recipe has stopped without success.
- Failing seeds: 500–1700 iterations and 10–30 s on the T tasks.
- Pen: about 220 ms per iteration alone on the GPU (380 with a second job running). The two pen runs took 302 s and 124 s.

### 1.6 Noise

Cycle 5 and cycle 6 of the hand-driven chain ran the same plan with the same flags on PushT2dHard seed 2. Cycle 5 accepted nothing; cycle 6 reached 0.0067. Default GPU ops are not deterministic, and a repeated cycle is a hidden re-roll. The same seed set gives different counts between the hand-driven chain (6/6 after six cycles, including that re-roll and my choice of level) and the automatic rule (4/6). Treat every count in §1.3 as ±1–2.

### 1.7 What remains unsolved after the pilot

| Class | Examples | Seen in |
|---|---|---|
| Near misses that creep at the fine level | 0.0101, 0.0157, 0.048, 0.052, 0.061 | T tasks; the pen's 524-iteration finish |
| Horizon used up | F5 PushT2dHard seed 0: stem−x [27–58], bar+y [76–140], ends at 0.240 while still pushing | T-hard |
| Stall at both coarse levels | PIP4F PushT2d seed 3 (0.1855), PLF PushT2dHard seed 0 (0.825) | T tasks |
| No contact ever | F5 PushT2d seeds 1, 3 | F5 only; PIP4F and PLF make contact on these seeds |

## 2. Diagnosis (interpretation of §1; not yet tested)

**D1. Discovery is needed per stroke, and the recipes provide it per solve.** After the first acceptance, the part of the horizon where the pusher is out of contact is planned with a model that has no signal there (`HANDOFF_trust_ratio.md` D3, applied locally in time).

**D2. The fine model is blind at the incumbent in two ways.** On an idle stretch there is no force and no gradient (T). At a graze the law has no slope at onset (pen, §17). A coarse law, possibly with the barrier, has a slope in both cases.

**D3. Choosing the level by trial is the main cost.** A wrong guess costs 255 iterations. Half of the coarse restarts are wrong guesses.

**D4. A global coarse restart also re-plans the strokes that already work.** The coarse model moves the early part of the plan, and the cycle is accepted only if the whole true rollout improves. I expect this to be a cause of unproductive cycles on the T and of the wild rollouts on the pen. Not measured.

**D5. Bursts converge slowly after an escape.** The pen needed 44 accepted fine bursts after the escape. LFDDP is the solver built for this phase (Jar: 5/5 in 6 s).

**D6. The pen needs no branch step.** The coarse restart is a model-derived, deterministic perturbation of the kind that §18.1 showed is enough (every random kick escaped the slide). The LPEC keeps a role as a certificate and as a fallback, not as the main mechanism.

## 3. Hypotheses

Each has a prediction that can fail. Thresholds are first choices, not tuned values.

| | Hypothesis | Prediction | Falsified if | Tested in |
|---|---|---|---|---|
| H1 | The cycle gain is a mechanism, not re-roll noise | Under deterministic ops, cycles from the 17 failing T plans and the pen slide improve the goal by ≥ 50 % on ≥ 9 of 18. With default ops, cycles beat a control that spends the same iterations on fresh runs from scratch and keeps the best: ≥ 2 more solved of 12 T-hard seeds, in two runs | The best-of-fresh-runs control solves as many seeds in both runs | E1 |
| H2 | The coarse smoothing is the active ingredient on the pen; the discovery barrier is needed in addition on the T | Pen: restart at level 1 with the post-acceptance barrier escapes the slide; restart at the fine level with the discovery barrier does not. T: level 1 without the discovery barrier is productive on < half as many plans as with it | The fine level with the discovery barrier is as productive as the coarse level (then the cycle is a barrier restart and smoothing can stay fine) | E2 |
| H3 | Most productive excursions show a true-goal gain early | With read-outs every 5 iterations, ≥ 80 % of productive coarse excursions have their first gain within 60 iterations | < 60 % (then a short cap loses too many cycles) | E3 |
| H4 | A capped excursion keeps the solve rate and cuts the waste | Cap and patience from E3: solved counts within 1 of the uncapped pilot on each T task, iterations in unproductive cycles ≤ 40 % of restart iterations (pilot 62 %) | Loses ≥ 2 seeds on a T task in two runs | E3, E4 |
| H5 | LFDDP converges an escaped incumbent much faster than bursts | Pen from the 0.197 plan: < 0.01 in ≤ 100 LFDDP iterations (bursts: 524). T near misses (0.0101–0.061): ≥ 4 of 6 below 0.01 | Pen needs > 300 iterations or stalls above 0.05 | E5 |
| H6 | Cycles inside PLF match or beat cycles on F5 | PLF with coarse cycles: PushT2d ≥ 11/12 and PushT2dHard ≥ 8/12 over two runs, pen < 0.01 in both runs | PushT2dHard ≤ 6/12 or pen fails twice | E4 |
| H7 | A computed blindness test can pick the level | On stored cycle-start incumbents, the predicted decrease of the candidate level separates productive from unproductive cycles with AUC ≥ 0.75 | AUC < 0.6 | E6 |
| H8 | Coarsening only the idle stretches is better than coarsening everything | Productive fraction of coarse cycles rises from about 0.5 to ≥ 0.7; no true rollout with goal > 10 on the pen; PushT2dHard gains ≥ 1 seed per run over H6's arm; states before the first coarse stage move by < 1 mm | Productive fraction does not rise and no seed is gained in two runs | E7 |
| H9 | The two-sided rule replaces the per-task barrier and starting-smoothing settings | With one setting (μ₀ 1e-3, level 1 start): PushSphere2d ≥ 9/10, PushBox2d ≥ 5/6, knob ≥ 17/20, fingers-hard ≥ 23/24 | Any of these drops by ≥ 2 seeds in two runs | E9 |
| H10 | Nothing here slows the solves that already succeed | Gate set: every task's median seconds per seed within +10 % of the base recipe, same solved count ±1; no seed takes more than twice the base budget in iterations | A task's median rises by > 10 % in two timing rounds | E10 |
| H11 | Where to coarsen can be read from the incumbent | Idle stretches plus the biactive stages of `bstat_probe.py` cover the stages that the successful cycles change (≥ 80 % of the stages where the plan's contact pattern changes) | Coverage < 50 % | E7 (logged), E11 |

## 4. Methods

Conventions as in `HANDOFF_trust_ratio.md` §4. All new code goes in `scripts/` here, standalone. Package edits invalidate the export cache (`~/.cache/mujodico_ippas/export`) and are left for last (§4.6).

### 4.1 The cycle as piloted

```
run F5 (from scratch or from a saved plan) until it stops
while not solved and cycles < NCYC:
    improved = goal < 0.9 * goal at the start of the last cycle
    if goal < 0.03:        level 0.1, post-acceptance barrier   (stop if already there and not improved)
    elif improved:         level 1, discovery barrier
    elif level 10 untried: level 10, discovery barrier
    else:                  stop
    run F5's burst loop from the incumbent at that level (k = 10, budget 250, same acceptance)
return the incumbent
```

Commands:

```bash
cd /home/gchou/PycharmProjects/mujodico-main-dev
H=mujodico/docs/handoff_2026-11-22
# vc.sh <Task> <a:b> <label> <ncyc> [extra c1_accept.py flags]; rows -> results/vcycle/vc.jsonl
$H/scripts/vc.sh PushT2dHard 0:6 V0 6 --mu0 0.1 --project lqr
$H/scripts/vc.sh PushT2dHard 0:6 VfromPIP4F 6 --mu0 0.1 --project lqr --from-plans $PWD/$H/results/hard/plans PIP4F
$H/scripts/vc.sh AllegroPen 0:1 VPEN 2 --mu0 1e-3 --from-plans $PWD/$H/results/hard/plans PLF
DET=1 $H/scripts/vc.sh ...        # deterministic XLA ops (equivalence tests only; about 2x slower)
```

`vc.sh` in `scripts/` is a copy of the session's script with its paths rewritten. It has not been run from there; run E0 first.

Known defects of the pilot rule, to fix before any rate comparison (E0):

1. A cycle that ends on its budget while still improving is treated like a stall. The pen was cut off at 0.197 this way.
2. Cycle-start incumbents are not saved. E6 needs them.
3. Checks happen only at burst ends.

### 4.2 Cycles inside PLF ("PLF-V")

PLF already has the structure: lifted IP exploration, LFDDP convergence, and a cycle back to the IP on exhaustion. Changes, all in `scripts/lfddp.py`:

1. New flags `--cycle-sig` (level of the cycle's exploration, default 0 = current behaviour), `--cycle-mu` and `--cycle-mumin` (its barrier; default the discovery values `--ip-mu`, `--ip-mumin`), `--cycle-cap` (default `--ip-cycle`), `--cycle-stall` (default `--ip-stall`).
2. In the `cycle` branch: call `explore(u_inc, Xr, Cr, sig_c, 0, cap)` with `Xr, Cr = restart_point(u_inc, sig_c, restart)`, i.e. the coarse level and no terminal boost. `explore` builds its parameters from `PHip`; pass the cycle's barrier by a copy of `PHip`.
3. After a productive exploration: reset the model ladder (`sig = sig_c`, `mu = mu0`, `boosts = 0`) and continue LFDDP from `start_point(Sip, sig_c, handoff_from)`. The accuracy test then sharpens as usual. After an unproductive one: keep the old `sig`, `mu`, `boosts` and count the cycle.
4. Level order per stall: 1, then 10 (H7 later replaces the order by a test). `--cycle N` remains the maximum number of cycles.
5. Record per cycle: level, iterations, incumbent before and after, and save the cycle-start incumbent (`us`) to `SAVE_DIR`.

`explore` already reads out the true goal every 5 iterations and stops on stall (20 iterations without a new incumbent). The pen's first accept came after 32–73 iterations, so the stall count is a real risk. E3 measures it before it is fixed.

### 4.3 Classifying a stop

Three cases, decided from numbers the driver already has:

| Case | Test | Action |
|---|---|---|
| Still improving | The last two accepted steps each gained ≥ 2 %, and the stop reason is the budget | Continue at the current level. Not a cycle |
| Near | Incumbent goal < 3 × success tolerance (0.03) | Finisher: LFDDP at the fine level from the incumbent's true states (§4.4) |
| Blind stall | Otherwise | Coarse cycle (§4.2), level from §4.5 |

### 4.4 Finisher

LFDDP from the incumbent with `--handoff-from true --restart true` semantics (true-law states with smoothed-law forces), fine level, post-acceptance barrier, PLF's own boosts. For E5, before PLF-V exists, start LFDDP from a saved plan with `--init-plan`: it needs an `npz` with `xs`, `us`, `cs`. Write `scripts/mk_init.py` (true rollout for `xs`, `S._forces` at the chosen level for `cs`; the pattern is `restart_point(..., 'true')` in `lfddp.py`).

### 4.5 Blindness test (picks the level without a trial)

- Quantity: LFDDP's expected decrease at the retraction of the incumbent under level σ and barrier μ, from one linearization and one backward pass (`S._lin`, then the backward pass; fingers-hard cost: 10–12 ms + 4–6 ms). Record both the total expected decrease and its terminal-cost part.
- Rule to test: the model at (σ, μ) is blind if the expected decrease is < κ_b · r* (κ_b = 0.25, mirroring the accuracy test's κ). Choose the finest level that is not blind.
- Together with the existing accuracy test this gives the two-sided rule: sharpen when the smoothing error at the incumbent exceeds κ · r*, coarsen when the model is blind.
- First use is offline (E6), on stored cycle-start incumbents with known outcomes.

### 4.6 Local fidelity (coarse only where the incumbent has no contact)

**Mask.** From the incumbent's true rollout: stage k is coarse if no contact force exceeds 1e-6 in stages k − d … k + d (d = 5), or if `bstat_probe.py` marks a biactive contact with a descent sign at k. All other stages stay fine.

**Step 1, LFDDP only (script-level, no cache invalidation).**

- `eps` becomes a vector of length N in `lin` (already vmapped over stages: change its `in_axes`), in `forward`, `rollout` and `_forces` (scans: carry `eps[k]`).
- `mu` becomes a vector of length N in `_lin`, `_fwd`, `_fwd_b`, `_cost` (the barrier is a per-contact, per-stage term already).
- A scalar is broadcast, so existing calls are unchanged.
- The cycle then runs LFDDP itself on the mixed model, with the discovery barrier on the coarse stages only. Cold LFDDP discovers contact on pushers with the barrier (PushSphere2d 6/6 at μ₀ 0.1) and is weak on fingers-hard (4/6), so this step is expected to work for the T and is uncertain for the pen.

**Step 2, per-stage smoothing in the lifted IP (package edit; only if step 1 is not enough).**

- `IpPasSolver._eq` passes `params[self._eq_param]` to every stage. Add an optional per-stage argument: in `_values` (the `jax.vmap(self._cons, in_axes=(0, 0, 0, None))` call), in the `stage` function of `_linearize`, and in `lag_e`. Grep for every `_eq(` and `stage_equality(` call first, including the condensed-contact path and the wrapper's `_make_stop_fn`.
- A scalar `contact_smoothing` must give bit-identical results to the present code (unit check under deterministic ops).
- The IP barrier parameter is global and stays global. The prefix is then protected only by the fine law, not by a missing barrier. Whether that is enough is part of E8.
- Do this after the last comparison that depends on the current export cache. Keep a copy of the cache directory.

The mask is not a window: the problem keeps its full horizon, its start state and its goal, and no intermediate target is introduced.

### 4.7 Speed budget

Rules that every arm must satisfy, and how each is checked:

| | Rule | Check |
|---|---|---|
| S1 | Cycles, the finisher and the blindness test run only after the base recipe has stopped without success | Under deterministic ops, rows of seeds solved by the base recipe are identical with and without the new flags |
| S2 | An unproductive excursion ends at the cap or stall rule of E3 | Iterations in unproductive cycles ≤ 40 % of restart iterations |
| S3 | Per seed: at most 3 cycles and at most 2× the base iteration budget in total | Maximum iterations per seed in every row |
| S4 | The blindness test costs at most one linearization and one backward pass per candidate level | Timed in E6 |
| S5 | Per-stage smoothing and barrier do not change the time per iteration | ms per iteration within 3 % of the scalar path, same process, alone on the GPU |
| S6 | Convergence after an escape uses LFDDP, not fine-level bursts | Pen: iterations after the escape (pilot: 524) |
| S7 | No extra compiled programs on the default path | Compile and cache-load time of the first seed within 10 % |

Timing protocol: one job on the GPU, warm caches, base and new arm back to back in the same session, medians over seeds with the first seed of each process excluded, two rounds. Report median, p90 and maximum seconds per seed, and for the hard tasks the time to the first r < 0.01.

## 5. Experiments

**Common protocol.** Start: rollout of zero controls unless a plan is given. Metric: `true_ok` (true goal < 0.01) on the returned plan, plus final goal, iterations, cycles, per-cycle level and outcome, rejected iterations, wall time. Rate comparisons: two runs per arm with default flags. Equivalence tests: deterministic ops. Rows go to `results/vcycle/` with one file per experiment.

**Failing-plan set** (inputs for E1, E2, E6): from `results/hard/plans/`, PushT2dHard F5 seeds 0, 3, 4; PIP4F seeds 0–4; PLF seeds 0, 1, 3, 4; PushT2d F5 seeds 0, 1, 3; PIP4F seed 3; PLF seed 0; and `PLF_AllegroPen_0.npz`. That is 17 T plans and the pen.

**Order of work.** E0 → E1 → E2 → E3, then E5 and E4, then E6, E7, and E8–E11 as their conditions are met. E10 closes every stage that changes a recipe. If only one day is available: E0, E1, E2, E5.

### E0. Reproduce and repair the pilot (two hours)

- Run the three commands of §4.1. Expect the counts of §1.3 within ±1 and the pen below 0.25 after the first cycle.
- Fix the three defects listed in §4.1 in `c1_vcycle.py`: continue instead of restarting when the budget ends during steady improvement; save cycle-start incumbents; nothing else.
- Rerun the pen with the fix. Pass: the pen reaches < 0.01 in one process without a hand-started continuation.

### E1. Is it a mechanism? (half a day; tests H1)

- Arm A: cycles (`NCYC=3`) from every plan of the failing-plan set, deterministic ops, one run.
- Arm B: the same with default ops, two runs.
- Control C: from scratch, F5 repeated until the same number of iterations as arm B's seed has been spent, keep the best plan. Two runs. This control is a random restart and exists only to measure what re-rolling alone buys.
- Report per seed: goal before, after A, after B (two values), after C (two values).
- Pen under arm A three times with default ops: count how often the slide is escaped.

### E2. Active ingredient (half a day; tests H2)

One restart cycle from each plan of the failing-plan set, deterministic ops, four arms:

| Arm | Level | Barrier |
|---|---|---|
| coarse + discovery (= pilot) | 1 | `mu0`, floor 1e-2 |
| coarse only | 1 | 1e-3, floor 1e-4 |
| barrier only | current fine level (0.001) | `mu0`, floor 1e-2 |
| neither (= existing restarts) | 0.001 | 1e-3, floor 1e-4 |

- Flags: `--ladder`, `--mu0`, `--disc-mumin`; the "near" branch of `c1_vcycle.py` already runs a restart with the post-acceptance barrier.
- Report per arm: productive cycles out of 18, median relative gain, wild rollouts.
- The outcome fixes what §4.2 and §4.6 have to vary: smoothing, barrier, or both.

### E3. When does a coarse excursion first pay off? (half a day; tests H3, H4)

- Rerun the coarse excursions of E1 arm A on the resumable solve (`scripts/ippas_resume.py`) with a true read-out every 5 iterations, up to 255 iterations, no early stop. Store the read-out stream.
- From the streams: the iteration of the first true-goal gain ≥ 2 % in productive excursions; the best gain reachable under a cap of 40, 60, 80, 120; the same under a stall rule of 20, 30, 40 iterations without a new best.
- Choose the cap and stall count that keep ≥ 90 % of the productive excursions' final gain at the lowest unproductive cost. Then rerun E1 arm B with them (H4).

### E4. PLF-V (one day; tests H6, H4)

- Implement §4.2 and §4.3 with E2's ingredient and E3's cap.
- Arms on PushT2d 0:6, PushT2dHard 0:6, AllegroPen, two runs each: PLF (base, §14.5 flags; T tasks as PushBox2d), PLF-V with `--cycle 3`.
- Compare with the pilot rows (`V0`, `VfromPLF`) and with E1 arm B.
- Report the per-cycle table (level, iterations, gain) and the split of time into IP, LFDDP, read-outs.

### E5. Finisher (two hours; tests H5)

- `mk_init.py` on `results/vcycle/plans/VPEN_AllegroPen_0.npz` (0.197) and on the near-miss T plans (`V0_PushT2dHard_4`, `V0_PushT2dHardLong_3`, `V0_PushT2dHardLong_4`, `VfromPIP4F_PushT2dHard_0`, `VfromPLF_PushT2dHard_4`, and the PIP4F PushT2dHard seed 1 base plan at 0.0143).
- LFDDP from each with `--init-plan`, fine level, `--lm 1 --par-ls 1 --ro-every 2`, PLF's boosts, no cycle.
- Report iterations and seconds to < 0.01, and the final goal.

### E6. Blindness test (one day; tests H7)

- Offline: for every stored cycle-start incumbent of E1 (arms A and B) compute the expected decrease of §4.5 at levels 0.001, 1 and 10, with and without the discovery barrier. Label each (incumbent, level) pair by whether its cycle was productive.
- Report ROC and AUC for the total expected decrease and for its terminal part, both divided by r*. Report the time per evaluation.
- If AUC ≥ 0.75: switch PLF-V's level order to the rule of §4.5 and rerun E4's PLF-V arm. Pass: solved counts not lower, rejected iterations lower.

### E7. Local fidelity in LFDDP (two days; tests H8, H11)

- Implement step 1 of §4.6. Unit checks, deterministic ops: a constant vector reproduces the scalar path to 1e-12 in `lin`, `forward`, `rollout` and the cost; ms per iteration within 3 % (S5).
- Arms on the failing-plan set (one cycle each), then on PushT2d, PushT2dHard and the pen (full runs, two each):
  - global coarse cycle (E4's arm);
  - local coarse cycle with LFDDP as explorer;
  - local mask with dilation d = 0, 5, 15.
- Log per cycle: the mask, the displacement of the states before the first coarse stage, the stages where the contact pattern changed (for H11), productive or not.

### E8. Per-stage smoothing in the lifted IP (two days; only if E7's local arm does not beat the global one and the IP explorer is the suspected reason)

- Implement step 2 of §4.6. Unit checks: scalar path bit-identical on fingers-hard seed 0 for 20 iterations; resumable solve still exact (`e2_resume_unit.py`).
- Arms as in E7 with the lifted IP as explorer on the mixed model.
- Rerun the gate's fingers-hard and PushSphere2d rows once to confirm the default path is unchanged.

### E9. One setting for all tasks (one day; tests H9; after E6)

- PLF-V with the two-sided rule and a single start (μ₀ 1e-3, level 1, no `--sig0`, no per-task `--ip-mu`): PushSphere2d, PushSphere2dHard, PushMultiSphere2d, PushBox2d, knob, fingers-hard, fingers, AllegroBallTableHard, two runs.
- Compare with PLF's rows in `results/lfddp/final.jsonl`.
- Keep `--handoff-from true --restart true` for PushBox2d and the T as now; this experiment is about the barrier and the starting smoothing only.

### E10. Gate (one day per adopted change; tests H10)

- Solve-rate gate: gate set (90 seeds) and knob (20), two runs. Adopt only if no task loses ≥ 2 seeds against PLF in both runs and fingers-hard is ≥ 23/24 in both.
- Speed gate: §4.7's protocol against PLF. Adopt only if every gate task's median is within +10 %, S1–S7 hold, and the hard tasks' time to solution is reported.
- Hard tasks, two runs: PushT2d, PushT2dHard, AllegroPen, AllegroJar, AllegroJarC (the Jar rows must not regress from 5/5).

### E11. LPEC as certificate and alternative escape (half a day; optional; after E2)

- At the pen slide and at four failing T plans: `bstat_probe.py` sign test, then three single cycles under deterministic ops: coarse restart, QP LPEC step with branches (`lpec.py --quad 1`, Δ 0.05) followed by LFDDP, and both in sequence.
- Report the goal after each. This ranks the two escapes on the pen and confirms that the LPEC does nothing on the T plans (§17).

## 6. Fallbacks

**F1. E1 shows the gain is re-roll noise (control C solves as many seeds).**

1. Check the pen separately: if the slide is escaped in ≥ 2 of 3 runs and under deterministic ops, keep the cycle for graze-type stalls only.
2. For the T, the conclusion is then that local restarts do not add to what a second independent solve adds. Report it, and bring the user the two options that the constraints had deferred: a mixed-integer step on the linearization with per-face candidates (extension of `lpec.py`), or batched GPU starts.

**F2. E2: neither ingredient alone works and both together are needed.** No change of plan; §4.6 must then carry both a per-stage smoothing and a per-stage barrier, which only LFDDP offers (step 1). If step 1 fails, the local idea is limited to smoothing in the IP with a global barrier (E8).

**F3. E2: the barrier alone is as good as the coarse level.** Drop the level change from the cycle. The cycle becomes "restart with the discovery barrier at the current level", which needs no ladder reset and no finisher. Rerun E1 arm B in that form.

**F4. E3: productive excursions pay off late (H3 fails).**

1. Use the cap that keeps 90 % of the gain even if it is 150 iterations, and limit cycles to 2 per seed (S3 still holds).
2. Replace the stall rule by the ratio test of recipe T on the read-out stream with its discovery guard (`HANDOFF_trust_ratio.md` §4.1), which ends an excursion on mistrust instead of on a count.

**F5. E4: PLF-V is worse than cycles on F5.**

1. Diff the two on the failing-plan set under deterministic ops: start point (`restart_point` mode), barrier values, boost reset.
2. If the IP exploration inside PLF is the weaker part (its `explore` has a 20-iteration stall), raise `--cycle-stall` to E3's value.
3. If still worse: keep `c1_vcycle.py` (with E3's cap) as the escape and use LFDDP only as the finisher of §4.4.

**F6. E5: LFDDP does not finish the pen or the near misses.**

1. Pen: try the hand-off at levels 0.01 and 0.1 with PLF's accuracy test doing the sharpening.
2. Near misses: check `diag` (plan goal at the fine level against true goal). Equal values mean a local minimum on the consistent trajectories; send those to a coarse cycle instead of the finisher (lower the "near" threshold to 0.015).
3. If neither: keep fine-level bursts with `--k 20` and accept S6's failure for the pen only; report its time.

**F7. E6: the expected decrease does not predict productivity (H7 fails).**

1. Try a short probe instead: 10 lifted IP iterations at the candidate level, and use the plan goal's decrease relative to r* as the score (cost: 10 iterations per candidate, against 255 for a wrong guess).
2. If that fails too: fixed order (1, then 10) with E3's cap. The cost is then bounded by two capped excursions per stall.

**F8. E7: local coarsening is not better than global.**

1. Larger dilation; a ramp of the smoothing over 5 stages at the mask's edges.
2. Suffix form: coarse from the first idle stage to the end of the horizon.
3. If the prefix still moves: start the retraction from the incumbent's true states (`--restart true`), which pins the prefix states at the start of the excursion.
4. If none helps, D4 is wrong. Keep global cycles and drop §4.6.

**F9. The package edit of E8 changes default results or breaks the cache.** Revert the edit (`git checkout` of `ippas.py` hunks only after confirming with the user, since the working tree has other uncommitted changes), restore the saved cache directory, and stay with step 1.

**F10. E9: one setting does not cover the pushers or the knob.** Keep the per-task `--ip-mu` and `--sig0` values of §14.5 and state that the two-sided rule unifies the cycle only.

**F11. E10: the speed gate fails.**

- If a median moved: S1 is violated somewhere. Find the seed whose row changed under deterministic ops and fix the trigger.
- If only p90 or the maximum moved: lower the cycle limit to 2 and the cap to the 80 % point of E3.
- Do not trade solved gate seeds for hard-task seeds.

**F12. The horizon-used-up and both-levels-stall classes remain (§1.7) after everything above.**

1. Acceptance on the true objective instead of the true goal (`--merit obj` exists in `c1_accept.py`): the running cost rewards reaching the goal earlier, which the goal-only test cannot see. One run on the T tasks.
2. Local coarsening of an early stretch that contains a stroke, to let a wasteful first stroke be re-planned (uses §4.6 unchanged, with a different mask).
3. If both fail, these seeds mark the limit of local methods on this task. Record the plans and bring the user the deferred options of F1 item 2.

## 7. Pitfalls already paid for

- **A budget stop is not a stall.** The pen was restarted at the coarse level while it was still improving at the fine one, and lost 500 iterations.
- **A coarse restart from a good incumbent can produce wild true rollouts** (goals of 10⁴ on the pen). Acceptance protects the incumbent; the iterations are still spent.
- **Identical reruns differ** (§1.6). Two runs per arm for rates; deterministic ops for any claim about a mechanism.
- **`--from-plans` is ignored by `c1_accept.py` when `--disc-mode none` is set** (the branch order). `c1_vcycle.py` fixes the order and accepts plans that hold only `us`.
- **The first seed of a process pays the cache load or compile** (10–90 s). A new horizon (PushT2dHardLong) compiles anew.
- **Two jobs on the GPU distort timing** (pen: 380 against 220 ms per iteration).
- **The pen's metric barely weights orientation** (0.01 against 100 for position). Report height and orientation error separately when claiming the pen.
- **Allegro links are visual-only** in the paper model (memory note of 2026-11-18): a pen success under the task metric does not exclude link penetration. Check clearance on any pen plan that is to be shown.
- **The user's constraints**: no windows, subgoals, tailored discovery phases, random sampling or sequence enumeration as fixes for now. The mask of §4.6 keeps the full horizon and the task's own goal.
- From `HANDOFF_trust_ratio.md` §7 and §10.9, still valid: judge on the true rollout only; CPU rollouts; never time with deterministic flags; kill jobs by PID.

## 8. What is not known

- Whether the pilot gain survives deterministic ops and the matched-budget control (E1).
- Which ingredient acts: smoothing, barrier, or the fresh duals that come with any restart (E2). On the pen the two barrier settings differ only in the floor, which acts when the solve is stationary; the slide is a stationary point, so the floor may matter there.
- When, inside an excursion, the true goal first improves (E3). The pilot checked at burst ends only.
- Whether unproductive cycles are caused by the disturbed prefix (D4) or by something else. No measurement yet.
- Whether LFDDP can explore on a mixed-fidelity model, or whether the infeasible lifted path is needed (E7 against E8).
- Whether the expected decrease means anything at a coarse level whose plans the true law does not follow (E6).
- Whether cycles ever fire on gate-set seeds that currently fail (fingers-hard seed 14 with F5) and what they do there.
- How much of the horizon-used-up class a longer horizon removes once cycles exist. The one PushT2dHardLong run gave 3/6 with two near misses, against 1/6 without cycles.
- Whether the yaw-weight sensitivity of §18.2 persists with cycles.

## 9. File map

| Path | What |
|---|---|
| `scripts/c1_vcycle.py` | Pilot: `c1_accept.py` plus the cycle loop (`NCYC`), plan loading fixed |
| `scripts/vc.sh` | Launcher with F5's flags and a per-seed summary; `DET=1` for deterministic ops. Path-rewritten copy, not yet run from here |
| `scripts/chk_pen.py` | True-law re-simulation of pen plans: goal, pen position, first lift |
| `scripts/strokes.py` | Contact strokes of a pushing plan along the true rollout |
| `scripts/bstat_probe.py`, `scripts/lpec.py` | Biactive sign test and LPEC step (§17, §18 of `HANDOFF_trust_ratio.md`) |
| `scripts/lfddp.py` | PLF; the `cycle` branch and `explore` are where §4.2 goes; `lin`, `forward`, `rollout` are where §4.6 step 1 goes |
| `scripts/pip.py`, `scripts/ippas_resume.py` | PIP4F; resumable IP solve used by `explore` and by E3 |
| `results/vcycle/vc.jsonl` | Pilot rows, automatic rule (labels `V0`, `VfromPIP4F`, `VfromPLF`, `VPEN`, `VPEN2`) |
| `results/vcycle/cyc.jsonl` | Hand-driven chain from the PIP4F PushT2dHard plans (labels `PIP4F_c1` … `PIP4F_c6w`) |
| `results/vcycle/plans/` | Plans of both; the pen solution is `VPEN2_AllegroPen_0.npz`, the 0.197 plan is `VPEN_AllegroPen_0.npz` |
| `results/vcycle/logs/` | Per-burst logs of the automatic-rule runs |
| `results/hard/plans/` | Base plans of F5, PIP4F, PLF on the hard tasks (the failing-plan set) |
| `../../trajopt/_src/fatrop_ref/ippas.py` | `_eq`, `_cons`, `_values`, `_linearize`: the call sites of §4.6 step 2 |
| `../../examples/methods/ippas_lifted_multiple_shooting.py` | `_params` (where `contact_smoothing` becomes a solver parameter) |
| To write | `scripts/mk_init.py` (E5), the `--cycle-*` flags and per-stage `eps`/`mu` in `lfddp.py` (E4, E7) |

## 10. Results (2026-11-23, night session): PLF-V, two-sided fidelity cycles inside PLF

Code: `scripts/lfddp_v.py` (a copy of `lfddp.py` with the cycle; `lfddp.py` itself is unchanged because another session's `lpec_loop.py` launches it). Launchers `scripts/vplf.sh` (PLF's §14.5 flags by task class, `SCRIPT=lfddp.py` for plain PLF), `scripts/vgate.sh` (gate + knob). Summaries `scripts/sumv.py`, `scripts/basevsfinal.py`, `scripts/pen_orient.py`. LPEC fallback driver `scripts/pv_lpec.py` (§10.11). Unused variant with §4.6 step 1: `scripts/lfddp_v_perstage.py`. The pilot driver `c1_vcycle.py` has E0's two fixes (budget-end extension while improving, cycle-start incumbents saved); the unfixed pilot is `c1_vcycle_pilot.py`. Rows in `results/vcycle/`: `plfv.jsonl` (T tasks), `plfv_pen.jsonl`, `plfv_near.jsonl`, `plfv_hard.jsonl` (Jar), `abl.jsonl` (ablations), `gate_v.jsonl`, `timing.jsonl`; plans in `results/vcycle/plans/`, logs in `results/vcycle/logs/`. No package source was modified.

**What was done from §5:** E0 (with both fixes), E1 (in a stronger form, §10.5), E2/E3 (through the excursion read-out streams, which every row now stores), E4 (PLF-V), E10 (gate, two runs; timing pair). §4.5 (blindness test) was replaced by a rule that needs no test (§10.2). §4.6 step 1 (per-stage smoothing and barrier vectors in LFDDP) is implemented in `scripts/lfddp_v_perstage.py` but was not needed and is not exercised (it perturbed PLF's path numerically, §10.7). E5–E9 and E11 were not run.

### 10.1 Summary

The GPU was shared with another session's pen jobs until about 01:00 and with several of my own jobs for most runs, so times are contended unless marked; rates are counted on the true law (goal < 0.01).

| Task | PLF (§14.4 / §15) | **PLF-V, final flags with 6 cycles** (per run; pooled; 10 cycles: §10.12) |
|---|---|---|
| PushT2d (6) | 5/6 | 6, 6, 6, 6 → **24/24** |
| PushT2dHard (6) | 2/6 (base inside this session's runs: 1–3 per run; seeds 0 and 4 never solved by it, 0/24 runs each) | 6, 6, 5, 5, 5, 5, and 5 under deterministic ops → **37/42** |
| PushT2dHardLong (6, N 220) | 0/6 (F5 1/6; F5 + pilot cycles 3/6) | 5, 6, 5 → **16/18** |
| AllegroPen (1 instance, from scratch) | 0/1 (slide, 0.597) | **14/19 runs** by PLF-V alone (6 or 10 cycles, one deterministic); stuck slides: **5 of 6** solved by the LPEC fallback of §10.11 (goals 0.0003–0.0078; pen lifted to 9.2–10.1 cm; link clearance ≥ −1 mm in all) |
| AllegroJar (5) | 5/5 | 5/5 (no cycle fired) |
| AllegroJarC (5, 3 mm margin) | 5/5 goal | 5/5 goal; 2/5 clear, identical to a PLF rerun tonight (§10.8) |
| Gate (90) | 90, 90 | **90, 90, 90** (G2, G3, G4) |
| Knob (20) | 19, 19 | **20, 20, 20**: the seed PLF always misses (10) is solved by one cycle |

- **No regression by construction, verified:** the cycle runs only after PLF has returned r* ≥ 0.01, and `lfddp_v.py` reproduces `lfddp.py` exactly when no cycle fires (PushBox2d 0:6: identical iterations, goals and times, §10.7). On the gate the cycle fired only on knob seed 10 (G2, G3, G4) and on fingers-hard seed 13 in G3 (a PLF re-roll miss, solved).
- Cost: seeds that need cycles take 7–95 s (PushT2dHard), up to 170 s (Long), 165–330 s (pen) on the contended GPU, where PLF returned a miss after 2–6 s (T) or 212 s (pen).

### 10.2 The method (PLF-V)

PLF (PIP → LFDDP with the sharpening ladder and its own fine-level cycles) is unchanged up to the point where it would return a failure. Then:

```
while r* >= 0.01 and cycles < 6:
    trigger:   PLF's ladder is exhausted (fine level, boosts and PLF's two fine cycles used) or LFDDP's budget ran out
    level:     sig_c = sig0 * 10^-j        j = consecutive cycles without a tau = 5 % decrease of r*  (reset to 0 after one)
               (no level >= sig_min left -> stop)
    start:     the incumbent re-simulated at sig_c if that reproduces r* within kappa r*, else its true states with sig_c forces
    excursion: lifted IP from the start, discovery barrier (PLF's --ip-mu / --ip-mumin)
               read-out every 5 its = best of the open-loop and the TVLQR closed-loop true-law rollout of the IP plan
               before the true law has responded: ends only at the cap (80 = the first exploration's cap)
               afterwards: ends on the ratio test (mistrust) or on 20 its without a tau-relative decrease, bounded by the budget
    then:      PLF's model ladder re-run from the new incumbent at sig_c (sharpen / boost as usual), with PLF's budget (400)
```

Each rule reuses a quantity PLF already has; the only new constant is the number of cycles, a budget bound (6 in most runs, 10 recommended, §10.12):

| Rule | Where it comes from |
|---|---|
| Level contracts ×10 per unproductive cycle, resets after a productive one | Trust-region radius update in model space; the ×10 of the sharpening ladder; τ = 0.05 of the ratio test |
| Start: simulation if it reproduces r* within κ r*, else the true states | PLF's accuracy test (κ = 0.25); `restart_point(…, 'auto')` already in `lfddp.py` |
| Closed-loop read-out of IP plans | LFDDP's own read-out is the best of open loop and DDP feedback; the IP plan has no gains, so TVLQR on the plan's linearization supplies them (F5's projection weights) |
| No stall count before the true law responds | The discovery guard of the ratio test, applied to the stall count |
| Stall = 20 its without a τ-relative decrease (after the first cycle) | τ and the stall count of PLF |
| Budget stop short of the goal triggers a cycle | §7's "a budget stop is not a stall", read the other way: it is not a success either |
| A cycle is PLF's ladder from the incumbent at σ_c, with PLF's budget | PLF itself |

The one story: **PLF manages the model fidelity in both directions.** The accuracy test sharpens when the model is wrong at the incumbent (existing); a fidelity cycle coarsens when the sharpest model is exhausted short of the goal, starting at the planning level and contracting toward the fine level as a trust region does after failed steps. Discovery is no longer one-shot: every cycle re-opens it from the incumbent.

Command (flags in `results/vcycle/V13flags.txt`; tonight's main runs used `V11flags.txt`, the same with 6 cycles):

```bash
cd /home/gchou/PycharmProjects/mujodico-main-dev
mujodico/docs/handoff_2026-11-22/scripts/vplf.sh <Task> <a:b> <label> <out.jsonl> \
    --vcycle 10 --v-budget 400 --v-guard 3 --ip-readout lqr --v-level tr --v-reset always \
    --prog-stall 1 --v-bud-cycle 1 --v-restart auto
# (most rows of §10.1–10.11 used --vcycle 6 = V11flags.txt; 10 is the recommended limit, §10.12)
# rows -> results/vcycle/<out.jsonl>, plans -> results/vcycle/plans/<label>_<Task>_<seed>.npz
# AllegroJarC: add --clear-margin 0.003 as for PLF
```

### 10.3 How the final form was reached (PushT2dHard 0:6 unless noted; one run per row; "base" = what PLF had returned in the same process before the first cycle)

| Run | Change | Base → final | What it showed |
|---|---|---|---|
| E0_V0 (F5 + pilot cycles) | reproduction of §1.3 | 3/6 (pilot 4/6) | within ±1 |
| E0_PEN | pilot + "continue while improving" fix | pen 0.597 → 0.061 in one process (pilot needed a hand-started continuation) | fix 1 works; the following level-1 restart from 0.061 wasted 255 its on wild rollouts (goals 5·10⁴) |
| PV1 | PLF-V: cycles at σ0, then 10σ0, PLF's explore as is | 3 → 3 | every coarse excursion ended at the 20-it stall before the true law responded |
| PV2 | + stall guard | 2 → 2 | excursions run to the cap but find nothing: open-loop read-outs of a long push see no progress |
| PV3 | + TVLQR read-out | 2 → 4 | every cycle productive (20–60 % per cycle); limited by the cycle count |
| PV4, PV4b | 6 cycles | 1 → 6, 2 → 5; PushT2d 5 → 6, 4 → 5 | remaining misses are near misses (0.034, 0.024) where σ0 and 10σ0 are too coarse |
| PNa (from the PV4b near misses, pen 0.0148) | level = coarsest level within κ r* | both T near misses solved; pen 0.0148 → 0.0029 | near misses need fine levels |
| PV5 | that rule for every cycle | 2 → 4 | far incumbents then get fine levels and creep (10–30 % per cycle) — negative |
| PV6, PV6b | trust-region level rule (σ0, contract ×10), ladder reset every cycle | 3 → 5, 3 → 5; PushT2d 6, 6 | misses: seed 4 hit the iteration budget |
| PV7 | cycle budget = PLF's (400) | 1 → 4 | pen: excursion still improving when the cap ended it |
| PV8 | cap binds only before the true law responds | 3 → 5 | pen: LFDDP after the escape crept at LM reg 1e4–1e5 for 350 its |
| PV9, PV9b | sufficient-decrease stall | 1 → 6, 2 → 4 | PV9b misses never reached a cycle (LFDDP budget ran out while creeping) |
| PV10, PV10b | budget end → cycle | 1 → 6, 1 → 5 | one miss from a far base (2.34) where no level responds |
| **PV11, PV11b** | **start point `auto`** (final) | **3 → 6, 2 → 6**; PushT2d 6 → 6, 5 → 6; pen solved in both runs | pen incumbents whose level-1 simulation diverges get the true states |
| PU, PUb | closed-loop read-out also in PLF's first exploration (`--ip-ro-scope all`) | 5/6, 5/6; pen 0/2 (0.18, 0.16) | negative: kept for cycle excursions only |
| R1, R2 (+ pen R1–R3) | final flags after reverting §4.6 step 1 | 5/6, 5/6; PushT2d 6/6; Long 5/6; pen 3/3 | reference rows for the final file |

### 10.4 Ablations of the final flags (PushT2dHard 0:6, one run each; full PLF-V 5–6 per run, 37/42 pooled)

| Removed ingredient | Solved | Misses |
|---|---|---|
| TVLQR read-out (open loop only) | 4/6 | 1.82, 0.24 |
| Cap binds after the response (`--v-guard 2`) | 5/6 | 0.15 |
| Sufficient-decrease stall (`--prog-stall 0`) | 5/6 | 0.014 |
| Budget end → cycle (`--v-bud-cycle 0`) | 6/6 | – |
| Ladder reset after every cycle (`--v-reset prod`) | 4/6 | 0.72, 0.014 |
| Trust-region level (`--v-level fixed`: σ0 then 10σ0) | 5/6 | 0.19 |
| Start `auto` (per-class `true` start) | 6/6, 5/6 (PV10, PV10b) | 2.34 |

Single runs with ±1 noise: the TVLQR read-out and the ladder reset are clearly needed on the T; the others each cost about one seed or matter for the pen (cap after response, sufficient-decrease stall, `auto` start; §10.6).

### 10.5 Mechanism, not re-roll (H1)

Over the 24 PushT2dHard runs of this session, base PLF (the same process before its first cycle) solved seed 0 in 0/24 runs and seed 4 in 0/24. With the final flags (6 runs) the cycles solve seed 0 in 5/6 runs and seed 4 in 4/6. Twenty-four independent re-rolls are a stronger control than E1's matched-budget best-of: re-rolling never solves these seeds. The cycle logs show the mechanism directly: e.g. seed 0 in PV11b, 1.954 → 1.714 → 1.136 → 0.852 → 0.737 → 0.183 → solved, each step one coarse excursion plus the ladder. Under deterministic XLA ops (PV11det) the cycles solve 5/6 (the miss, seed 3, is a base plan with an 80-step idle start from which no level responds), so the gain is not GPU nondeterminism either.

Excursion statistics (PV9–PV11, 81 coarse excursions on the T): 60 productive; 1680 of 3710 excursion iterations (45 %) were spent in unproductive ones (pilot 62 %). By level: σ0 productive in 57/69, finer levels in 3/12 (they are reached only at the end of a run).

### 10.6 AllegroPen

- **From scratch, final flags: 7 of the first 9 runs solve the pen** (PV11, PV11b, PV11e, PV11f, R1, R2, R3; goals 0.0003–0.0078; with the end-to-end runs of §10.11, 10 of 13). Each needed exactly one fidelity cycle: PLF ends on the slide (0.597); the σ0 excursion runs 100–390 iterations because the true goal keeps improving (0.597 → 0.035–0.31), then PLF's ladder finishes. Under the true law the pen is lifted to 9.2–10.1 cm (first lift at steps 68–71), position error 0.3–8 mm, link clearance ≥ −1 mm in every solution (`eval_hard.py`), orientation error 15–45° (from 90°; the task metric weights orientation 0.01, so the criterion certifies the lift, not the uprighting). Wall time 165–290 s per run, contended. Plans: `results/vcycle/plans/{PV11,PV11b,PV11e,PV11f,R1,R2,R3}_AllegroPen_0.npz`.
- **The 2 failures** (PV11c, PV11d) end on a slide from which no excursion level responds within the 80-iteration discovery cap: the level-1 plan goal itself stays at 0.595. Tried from those slides (single runs): start points `true`, `track`, `lifted` (none escapes), the trust region starting at 10σ0 (no), a cap of 160 (escapes and solves PV11c's slide, 0.0020; not PV11d's), an LPEC branch step (`lpec.py --quad 1`, Δ 0.05) followed by PLF-V (PV11c's slide solved 0.0034, PV11d's to 0.080, PV2's back to the slide).
- **Why the outcome is slide-dependent:** the smoothed simulation of a slide is chaotic. The same controls give a level-1 plan goal of 0.6, 31 or 4647 after tiny LFDDP changes that leave the true goal at 0.5972 (`accs` in `cyc_log`). The `auto` start (simulation only if it reproduces r* within κ r*) removed the failures caused by a diverged start (PV8 → PA8: solved); what remains are slides where the coarse model has no descent at all.
- Earlier variants, for reference: from §1.3's saved PLF slide plan, PLF-V reached 0.0148 (closed-loop read-out) and 0.0453 (open loop); from 0.0148 one fine-level cycle gave 0.0029 (PNa). The pilot's F5 cycles with the budget fix: 0.061 in one process (E0_PEN).

### 10.7 Timing

Timing pair, PLF (`lfddp.py`) and PLF-V back to back, same session, GPU otherwise idle except for the other session's occasional job (median s per seed, first seed excluded):

| Task | PLF | PLF-V |
|---|---|---|
| fingers (24) | 0.74 | 0.75 |
| fingers-hard (24) | 2.75 (23/24) | 2.66 (24/24; one cycle fired on PLF's miss) |
| PushBox2d (6) | 1.13 | 1.10 (identical iterations and goals on every seed) |

- The first timing pair showed PushBox2d +22 % with no cycle firing (1.44 vs 1.18 s; seed 4 took 148 instead of 64 iterations). Cause: §4.6 step 1 (per-stage smoothing/barrier vectors in LFDDP) is numerically equivalent but compiles differently, and the solver path diverged. It was reverted (kept in `scripts/lfddp_v_perstage.py`); since then `lfddp_v.py` reproduces `lfddp.py` to the iteration where no cycle fires (table above, PushBox2d run after the revert). The fingers rows are from before the revert. Rows labelled R1–R3 and G4 use the reverted file; earlier PV11* rows and G2/G3 used the per-stage file (same method, numerically perturbed base path).
- PLF-V compiles the lifted IP with a 400-iteration cap (so a responding excursion can continue) instead of 90; this sizes the filter only and does not change the path or the time (PushBox2d check).
- Seeds that need cycles: PushT2dHard 7–95 s (contended), PushT2dHardLong up to 170 s, pen 200–330 s per run (the pen alone runs at ~0.25 s per IP iteration).
- **Idle GPU, nothing else running** (`results/vcycle/idle.jsonl`, PLF then PLF-V, back to back; first seed includes ~30 s compile):

| Task | PLF (solved; s per seed) | PLF-V (solved; s per seed) |
|---|---|---|
| PushT2d 0:6 | 5/6; 35.4, 3.0, 1.6, 2.9, 1.2, 2.0 | 6/6; 40.0 (one cycle), 3.0, 0.9, 2.8, 0.6, 1.9 |
| PushT2dHard 0:6 | 2/6; 34.9, 4.6, 1.6, 3.3, 4.5, 1.9 | 5/6; 50.8, 7.5, 1.2, 6.0, 20.2 (miss, 6 cycles), 1.9 |
| AllegroPen | (§15: 0/1) | 1/1; 136 s |

  Where cycles are needed they cost 1.5–4× PLF's time to its miss on the T (6–20 s), and the pen is solved in 136 s.

### 10.8 Negative or neutral results

- **Accuracy-chosen level** (coarsest level whose smoothing error at the incumbent is within κ r*): solves near misses at once but gives far T incumbents fine levels that creep (PV5 4/6). The trust-region rule reaches fine levels only after coarse ones fail.
- **Fixed levels σ0, 10σ0** (the pilot's order): leaves near misses (PV4b 0.034, 0.024).
- **Pen start points for stuck slides:** true states + coarse forces (`true`), TVLQR tracking of the incumbent under the coarse law (`track`/`autotrack`), the incumbent itself in the lifted variables with its true contact forces (`lifted`, dynamics exact to 4e-16), starting the trust region at 10σ0 (`--v-top 1`): none escapes the slides on which no level responds (§10.6). A discovery cap of 160 escaped one of two.
- **AllegroJarC clearance:** PLF-V and a PLF rerun tonight (`lfddp.py`, same 3 mm margin) give the same plans and 2/5 clear (−2, −5, −4 mm), against 5/5 for §15's single PLFm3 run. On these seeds LFDDP reaches the goal stop tolerance after 6 iterations, before the clearance penalty acts; clearance is not part of PLF's stop test. Pre-existing, not caused by the cycle.

### 10.9 Pitfalls added

- **The coarse simulation of a stuck contact plan is chaotic.** The same pen slide controls give a level-1 plan goal of 0.6 or 31 or 4647 depending on tiny control changes that leave the true goal at 0.5972. Any rule that starts from the coarse simulation has to check it against r* (`auto`).
- **Open-loop read-outs hide coarse progress on long pushes;** read out closed loop.
- **Too many Allegro jobs on the 16 GB GPU run out of memory**, including other sessions' jobs: at about 00:45 six of my pen jobs made the other session's `lpec_loop.py` seed-2 continuation (rounds 6–7) fail with CUDA OOM ("lifted nan", "resolve failed"). Keep ≤ 3 Allegro jobs.
- `pkill -f <pattern>` matches the invoking shell when the pattern appears in its own command line (it killed my shell); kill by PID.

### 10.10 What is not known / next

- The pen from slides on which no smoothed model responds (2 of 9 runs): solved by an LPEC branch step + PLF-V on every such slide tried (§10.11, three slides). Not yet folded into `lfddp_v.py` itself (the driver calls it from outside); not tried on the T, where §17 found no useful branch.
- The T's remaining class: near misses at 0.04 on the long horizon where no level responds (PushT2dHardLong seed 4).
- One rule for every exploration: the closed-loop read-out in PLF's first exploration was negative (PU, PUb); the stall guard there is untested.
- §4.6 local fidelity: step 1 is implemented in `scripts/lfddp_v_perstage.py` (not in the main file, see §10.7) and untested. The remaining T misses look like its target: PV11det seed 3 pushes once, at steps 80–106, after an 80-step idle start; PU seed 4 used up the horizon on bar strokes (solved plans of both seeds switch to the stem).
- The cycle limit (6) bound productive runs (R2 seed 0: every cycle productive, 1.954 → 0.141); resolved by 10 cycles, §10.12.
- Deterministic-ops gate and a timing round on an idle GPU.

### 10.11 Addendum: LPEC branch step where no smoothed level responds (pen)

The two pen failures of §10.6 end on slides that are stationary for every smoothed model PLF-V tries. That is the case the §17 sign test flags as non-B-stationary (fingertip grazes with negative multipliers), and the LPEC branch step of §18.1 is the matching local tool. Composition, without any pen-specific rule:

```
incumbent = PLF-V result
while it misses the goal:
    LPEC branch step at the incumbent (lpec.py --quad 1, trust radius D; no biactive index -> stop)
    PLF-V from the step's true rollout (--from-us: its first excursion starts at the step)
    accept if the true goal improves (keep D), else D /= 2.5 (at most 2 shrinks in a row)
```

Driver: `scripts/pv_lpec.py <Task> <seed> <outdir> [--from plan.npz]` (rows `results/vcycle/pv_lpec.jsonl`, summaries `results/vcycle/pvl/`). It has the same structure as the other session's `lpec_loop.py`, with PLF-V instead of a single LFDDP re-solve.

| Stuck slide (from) | Steps | Result |
|---|---|---|
| PV11c | LPEC Δ 0.05 → PLF-V | **0.0034** (hand-run, LPV_PV11c) |
| PV11d | LPEC Δ 0.05 → PLF-V 0.080 → PLF-V continued | **0.0030** (hand-run, LPV_PV11d, LPV2_PV11d) |
| PV2 | LPEC Δ 0.05 → PLF-V falls back to the slide; LPEC Δ 0.02 → PLF-V | **0.0025** (hand-run, LPV_PV2, LPVs_PV2) |
| PV11d (driver, end to end) | PLF-V from the slide: 0.597 (stuck); LPEC Δ 0.05 (29 biactive, 11 pressed) → PLF-V | **0.0062** in 556 s (`pvl/PVL_PV11d.log`) |
| PV2 (driver, end to end) | PLF-V from the slide: 0.597 (stuck); LPEC Δ 0.05 (24 biactive, 6 pressed) → PLF-V | **0.0028** in 464 s (`pvl/PVL_PV2.log`) |

All five solutions keep link clearance ≥ −1 mm; the two driver solutions have position error 7.2 and 2.3 mm and orientation error 38° and 56°.

With the branch step as the fallback, every pen slide seen tonight was solved: 7 of the first 9 runs by PLF-V alone, and the other 2 slides plus PV2's (a slide from an earlier PLF-V variant) by LPEC + PLF-V (PV11d and PV2 also end to end with the driver, one LPEC round each). Caveat: three slides, one run each, and the LPEC step at Δ 0.05 fell back to the slide once (PV2, hand-run), so the shrink matters. The LPEC adds about 45 s (Gurobi, CPU) per step.

**End to end from scratch (driver, `pvl/E2E1–4.log`, two runs at a time): 4/4 solved.** Three by PLF-V alone (goals 0.0014–0.0025, 151–178 s). In one (E2E1), PLF-V ended on a stuck slide: the LPEC step at Δ 0.05 was rejected (the re-solve returned to the slide), and at Δ 0.02 it was accepted → 0.0023 (634 s in total). All four solutions: link clearance ≥ −1 mm, position error 2–3 mm, orientation error 35–47°.

Counting every from-scratch pen run with the final method tonight: PLF-V alone 10/13 (PV11, b, e, f, R1–R3, E2E2–4), and with the LPEC fallback every stuck slide encountered was solved (4 of 4: PV11c, PV11d, PV2, E2E1).

**Deterministic XLA ops** (`DET=1`, about 2× slower): PushT2d 6/6 (PDET). The pen from scratch (PDET) ended on a stuck slide (no level responds), and the driver from that slide under deterministic ops (`pvl/DETL.log`) solved it: LPEC Δ 0.05 → PLF-V 0.149 (accepted) → LPEC Δ 0.05 → 0.166 (rejected) → LPEC Δ 0.02 → PLF-V **0.0029** (1599 s in total). The whole pen pipeline is therefore reproducible without GPU nondeterminism. In round 1 PLF-V used all 6 cycles while every cycle was still productive (0.597 → 0.149), as in R2 seed 0 on the T: if the cycle limit is changed, raise it (10 was neutral on the T, §10.10) rather than lower it.

### 10.12 Recommended setting: 10 cycles (`results/vcycle/V13flags.txt`)

Three runs above stopped at the 6-cycle limit while every cycle was still productive: R2 seed 0 (T-hard), DETL round 1 (pen), and R1 seed 4 (T-hard), which a fresh PLF-V run from its 0.203 plan then solved in 60 s without any LPEC. The cycle limit is a budget bound, and a larger one only adds iterations to runs that would otherwise fail. With `--vcycle 10` (all else as V11):

| Task | Runs | Solved |
|---|---|---|
| PushT2dHard | C10a, C10b, C10c, C10d | 6, 5, 6, 6 → **23/24** |
| PushT2dHardLong | C10 | **6/6** (first solve of seed 4) |
| PushT2d | C10 | **6/6** |
| AllegroPen (from scratch) | C10a, C10b, E2E5, E2E6 | **3/4** by PLF-V alone (0.0012, 0.0012, 0.0020; E2E6 stuck, see the end of §10.11) |
| Gate (90) + knob (20) | G5 | **90/90, 20/20** (the cycle fired only on knob seed 10) |

Also: at the stuck deterministic T incumbent (PV11det seed 3, 0.279) the LPEC finds no biactive index (B-stationary), so `pv_lpec.py` stops at once, as §17 predicts: the branch step is the fallback for graze-type stalls (pen), not for the T. `pv_lpec.py` now uses these flags by default (`VFLAGS` env to override).

**Last two end-to-end runs with the recommended flags (10 cycles; `pvl/E2E5.log`, `E2E6.log`):** E2E5 solved by PLF-V alone (0.0020, 195 s). E2E6 stuck on a slide; LPEC rounds 0.597 → 0.464 → 0.403 (accepted), 0.482 (rejected, Δ 0.05), then Δ 0.02 → **0.0396**, ending at the driver's 4-round limit while its last PLF-V call was still improving through all 10 cycles (0.165 → 0.041); 1487 s. Unsolved within the limits.

**Pen totals, all from-scratch runs tonight with the final method:** PLF-V alone solved 14 of 19 runs (6 cycles: 11/15 including one deterministic run; 10 cycles: 3/4). On the 6 stuck slides the LPEC fallback solved 5 (PV11c, PV11d, PV2, E2E1, the deterministic PDET), and E2E6 reached 0.0396.

## 11. Results (2026-11-23, day session): LPEC folded in, local fidelity, speed

Asked: fold the LPEC fallback into `lfddp_v.py`; try coarse-only-on-idle-stretches (§4.6); speedups (1) read-out overlap, (2) per-process start-up, (3) parallel runs for small tasks (only once certain), (4) Gauss–Newton dynamics Hessian in the lifted IP, (6) profile the pen iteration — in order of fruitfulness. All in `scripts/` (no package edit; the package export cache is untouched). `lfddp_v.py` with the new flags off is the night's PLF-V (pre-edit copy `scripts/lfddp_v_pre1123.py`); the one default change, the export of the IP loop (§11.4), gives bit-identical iterates.

### 11.1 Summary

**Recommended setting: `results/vcycle/V14flags.txt` = V13 + `--ip-hess-cycle 0 --lpec 6`** (GN dynamics Hessian in the fidelity-cycle excursions; in-process LPEC fallback, at most 6 rounds). One flag set for every task, as before.

Final evaluation of V14 (§11.8; one run per seed, new seeds beyond the night's 0–5 included):

| Task | V14 | V13 (exact Hessian, no LPEC) on the same seeds |
|---|---|---|
| PushT2dHard 0–29 | **29/30** (LPEC fired twice: seed 0 solved, seed 4 not) | 5/6 + 22/24 = 27/30 (EX1, EXn) |
| PushT2d 0–17 | **18/18** (LPEC solved seed 11) | 6/6 + 11/12 = 17/18 (C10, EXn) |
| PushT2dHardLong 0–17 | **17/18** (LPEC solved seed 11; seed 0 B-stationary at 0.61) | 6/6 + 11/12 = 17/18 |
| AllegroPen from scratch | **5/5** (4 by PLF-V alone in 70–108 s; one LPEC rescue, 738 s); with §11.5's GC1–7: PLF-V alone 11/12 | 15/21 by PLF-V alone, 170–320 s |
| Gate 90 + knob 20 | **90/90, 20/20** (GC1; no miss, so the LPEC never runs; identical path to PLF wherever no cycle fires) | 90/90, 20/20 |

- **Speed:** the pen in 70–131 s when PLF-V solves it (V13: 170–320 s); GN cuts the coarse excursion's IP iteration by 45–61 % on Allegro (pen 139 → 54 ms, JarC 58 → 32 ms) and is neutral on the launch-bound 2-D tasks; the IP loop's export saves ~17 s (T) / ~25 s (pen) of start-up per process with identical iterates; split seed ranges run 1.6–1.65× faster in 3 GPU or 4 CPU processes (`vpar.sh`).
- **LPEC inside PLF-V:** the stuck pen slide PV11d → 0.0018 in 253 s (the outer driver: 556 s). On the T, the LPEC step also freed the stuck incumbent in all 5 misses on which it ran (§11.6); a restart without the step freed 1 of the same 5 (control).
- **Negative:** coarse only on idle stretches as LFDDP on a mixed model (§11.3); the read-out overlap (exact, no gain, §11.4); GN in every IP call (perturbs PLF's base path: PushBox2d seed 5 at 0.0091, pen 4/9, §11.5).
- **Caveat:** the LPEC-rescued pen plan V14p2 ends with the pen flipped (orientation error 133° from the goal; position error 0.2 mm). The task metric weights orientation 0.01, so it counts as solved, as every pen solution here; the other V14 pen plans end at 20–34°.

### 11.2 LPEC branch step inside PLF-V (`--lpec R`)

`scripts/lpec_step.py` is `lpec.py --quad 1` as a function (gurobipy in-process, same linearization, same MILP branch choice → convex QP with the binaries fixed, same constants `GAP_TOL 1e-3, F_TOL 1e-6, M_F 5, M_G 0.05`). `lfddp_v.py` wraps its solve in rounds:

```
round 0: PLF-V (as before)
while the best plan misses the goal and rounds < R:
    LPEC step at the best plan, trust radius D (no biactive index -> stop: B-stationary)
    round r: PLF-V from the step's controls (its first exploration is a coarse excursion, as --from-us)
    the best plan of all rounds is the incumbent (monotone);
    a round without tau-decrease of the best goal shrinks D by 2.5 (third shrink in a row -> stop)
```

The driver's rule (`pv_lpec.py`) with τ (PLF's own sufficient-decrease constant) as the acceptance test instead of 1e-4. One process: no re-tracing, no re-compiling, no plan files. Row field `lpec` lists the rounds.

| Test | Result |
|---|---|
| From the stuck slide PV11d (0.597), `--lpec 4` | round 0 PLF-V 0.597 (stuck); LPEC Δ 0.05: 29 biactive, 11 pressed (as the driver), step goal 0.820, 32 s; round 1 PLF-V → **0.0018**; **253 s** total (driver end to end: 0.0062 in 556 s) |

### 11.3 Coarse only on idle stretches (`--v-local 1`): negative

§4.6 step 1: the cycle's excursion is LFDDP on a mixed model — smoothing σ_c and the discovery barrier μ0 on the stages where the incumbent's true rollout has no contact force (normal force < 1e-6, the LPEC's threshold; no dilation constant), the fine level elsewhere (per-stage vectors through `lfddp_v_perstage.LFDDP`, used only in this excursion so the base path is unchanged). Read-outs, ratio test, stall and cap as in the coarse IP excursion.

PushT2dHard seed 4 (the exact-Hessian holdout): 0.41 (sim start) and 1.63 ('auto' start, as `cycle_start`), against 0.09–0.20 for the IP excursion. The mixed-model LFDDP creeps (stall after ~20 iterations, a few % decrease) or converges in 1–16 iterations; it does not discover a new face. Consistent with §4.6's caveat (cold LFDDP discovers weakly) — the discovery is in the lifted IP. The per-stage smoothing in the IP (§4.6 step 2) needs the package edit and was not attempted. Flag kept, off.

### 11.4 Speed

**Where the pen's time went** (idle GPU):

| Piece | Cost |
|---|---|
| Lifted IP iteration, exact Hessian (`ip_bench.py`, 60 its from the zero-control start) | 139 ms |
| … Gauss–Newton on the dynamics (`--ip-hess 0`) | **54 ms** |
| AllegroJarC: exact / GN | 58 / 32 ms |
| PushT2dHard: exact / GN | 12.8 / 12.6 ms (launch-bound) |
| Read-out per 5-iteration chunk (`ro_bench.py`): lin (GPU) 48 ms, TVLQR 11 ms, open + closed-loop true rollouts 54 ms (contact-free plan; ~175 ms on contact-rich plans) | |
| Process start-up of the resumable IP loop: `HS._init` trace + compile 7.7 s (T) / 12 s (pen), `HS._loop` trace 11.5 s / 17.5 s, compile from the persistent cache 7 s / ~10 s; LFDDP `_lin` first call 4 s / 32 s | |

The night's "245 ms per pen IP iteration" was mostly per-process start-up booked to the IP timer (and read-outs).

**(6) Profile of the GN pen iteration** (`ip_bench.py --profile`, `trace_sum.py`): GPU busy 49 ms/it of a 60 ms/it span (GPU-bound, not launch-bound): command buffers (linearization and fused ops) 26 ms/it, `triangular-solve` 16 ms/it in ~460 calls/it (the per-stage Riccati recursion of the KKT factorization, sequential over 200 stages, ~35 µs per call), Cholesky 1.8, LU 1.2. The next gain would need the package's KKT solver (fused per-stage kernels or a parallel-in-time Riccati); not attempted.
  - **Correction (later the same day):** the triangular solves are not the Riccati recursion. They are batched over the 200 stages with 22×22 factors (the Allegro mass matrix) and up to 1782 = 81·22 right-hand sides: forward-mode Jacobians differentiating the Cholesky factorization in `forward.implicit` and in the SAP context. Fixed by `small_la.spd_solve` (custom derivative `M⁻¹(db − dM x)` with `M⁻¹` applied as a product) and a gradient-stopped SAP factor (only used by stop-gradient Delassus estimates and the initial guess): pen IP iteration 54.7 → 45.9 ms (GN), 142 → 129 ms (base), LFDDP linearization 39 → 28 ms, Jacobians equal to 5e-13; T bit-identical (n ≤ 16 keeps the old path). See §12.

**(2) Export of the resumable IP loop** (`scripts/hs_export.py`, on by default; `PLF_HS_EXPORT=0` disables): `HomotopyIpPas._init`/`_loop` serialized with `jax.export` on first use, keyed on the package fingerprint (`solver.export_key`), the source of `homotopy.py`/`ippas_resume.py`/`hs_export.py`, argument shapes and first-call weak types, JAX version and backend. Bit-identical iterates under deterministic ops (402/402 state leaves after 15 iterations, PushT2dHard). Start-up of the IP loop: PushT2dHard ~26 s → 9.3 s, pen ~40 s → 14 s (warm export; the first process per task and flag set pays ~60–70 s once).

**(1) Read-out overlap** (`--ro-overlap 1`): the next IP chunk is dispatched before the CPU part of the read-out (TVLQR and the true rollouts). Exact (PushT2dHard 0:3 under deterministic ops: identical goals and iteration counts). No gain: PushT2dHard none; pen (GN) 128–130 vs 124 ms per IP iteration on the same slide paths. The GPU half of the read-out (the linearization, ~350 ms per chunk on contact-rich plans) must precede the dispatch, and the CPU rollouts did not overlap the chunk in practice (the GPU while loops need the host). Not adopted.

**Line search:** sequential step sizes (exact, same accepted step) would cost 2.7 rollouts on average on the pen (accepted α spread 0.25/0.25/0.18/0.18/0.14 over 1…0.03) — no gain over the 5 concurrent CPU rollouts.

(4): §11.5; (3): §11.7.

### 11.5 (4) Gauss–Newton dynamics Hessian: in the cycle excursions only (`--ip-hess-cycle 0`)

`--ip-hess 0` sets the lifted IP's dynamics-curvature weight to 0 (`hess=[0, 0, 0]`, skipped by `lax.cond` in the package; runtime parameter, no recompile). `--ip-hess-cycle 0` does it only in the fidelity-cycle excursions (and the first excursion of a `--from-us`/LPEC round, which is one); PLF's own first exploration and fine cycles keep the exact Hessian, so **every seed on which no fidelity cycle fires runs PLF's path unchanged.**

| | exact (V13) | GN everywhere (`--ip-hess 0`) | GN in cycles (`--ip-hess-cycle 0`) |
|---|---|---|---|
| Gate 90 + knob 20 | 90/90, 20/20 (G4, G5) | 90/90, 20/20 — but PushBox2d seed 5 ends at **0.0091** after 5 cycles, 18 s (exact: 1 s), reproduced (GN1, GN2); seed 4 needs a cycle | **90/90, 20/20**, times as exact (one knob seed fires a cycle) |
| PushT2dHard 0–5 | 5/6 (EX1); 23/24 (C10a–d) | 6/6 ×3 (GN1–3) | 4/6, 4/6, 5/6 (GC1–3; seed 0 stops at 1.3075 in all three: no excursion responds) |
| PushT2dHard 6–29 (new seeds, one run) | 22/24, 257 s | 11/12 (6–17) | 22/24, 240 s |
| PushT2d 0–5 / 6–17 | 6/6 (C10) / 11/12 | 6/6 / – | 6/6 / 10/12 |
| PushT2dHardLong 0–5 / 6–17 | 6/6 (C10) / 11/12 | 6/6 / 5/6 (6–11) | 6/6 / 10/12 |
| AllegroPen from scratch | 14/19 (night) + 1/2 today = 15/21; 170–320 s | 4/9 (GN1–6, OVG1–2, NOG1); 100–230 s | **7/7** (GC1–7), **80–131 s** |
| AllegroJar / JarC | 5/5, 5/5 | 5/5 (1.7 s median), 5/5 (2.8 s; 2/5 clear, as exact §10.8) | – (no cycles fire: = exact) |

Reading: on the T family the three are within seed noise (new seeds: exact 44/48, GN-in-cycles 42/48, different seeds fail); GN everywhere perturbs PLF's base path (PushBox2d, pen 4/9). GN in the cycle excursions only keeps the base path and makes the pen's coarse excursions both cheaper (54 vs 139 ms per IP iteration) and more effective (7/7 vs ~70 %). The exact curvature of a coarse smoothed model is no more faithful to the true law than its Gauss–Newton approximation; GN is the cheaper, positive-semidefinite choice for the exploratory model, and no constant is added.

Pen plans (GC1–7 and the two LPEC rounds below): link clearance ≥ the margin in all (`eval_hard.py`), position error 1.0–8.7 mm, orientation error 23–59° (from 90°; the task metric weights orientation 0.01, as §10.6).

### 11.6 LPEC fallback on the T (correction of §10.12's last paragraph)

With GN in the cycles and `--lpec 6`, the 6 seeds missed in §11.5's new-seed runs: **6/6 solved** (`results/vcycle/lpec_t.jsonl`, label LP14):

| Seed | Missed at | LPEC rounds (biactive, pressed, step goal → round result) | Final, time |
|---|---|---|---|
| PushT2dHard 23 | 0.206 | none needed this run (round 0 solved) | 0.0026, 26 s |
| PushT2dHard 25 | 0.545 | Δ 0.05 (1, 0, 1.18 → 0.0022) | 0.0022, 67 s |
| PushT2d 10 | 0.017 | Δ 0.05 (8, 0 → 0.024, rejected); Δ 0.02 (8, 0 → 0.0026) | 0.0026, 73 s |
| PushT2d 11 | 0.210 | Δ 0.05 (167, 1 → 0.0005) | 0.0005, 37 s |
| PushT2dHardLong 10 | 0.285 | Δ 0.05 (1, 0 → 0.389, rejected); Δ 0.02 (1, 0 → 0.0054) | 0.0054, 91 s |
| PushT2dHardLong 11 | 0.0197 (all three variants) | Δ 0.05 (10, 5 → 0.0010) | 0.0010, 71 s |

**Control** (`CTRL`): PLF-V restarted from the same stuck plans without the LPEC step stays on 4 of 5 (every coarse excursion capped, no response, at all four levels); PushT2d 11 solves by re-roll. So the step, not the restart, frees them. In three of the five the step presses no branch: it is then the true-law trust-region QP step (second-order task model, complementarity kept), which moves the plan off a point where every smoothed model is stationary even though its own true goal is worse (e.g. 0.545 → 1.18); PLF-V from there solves. §10.12's "B-stationary at the stuck T incumbent" held for that incumbent (PV11det seed 3) but not in general. Each LPEC step on the T costs 5–9 s (Gurobi; the linearization's first call compiles).

### 11.7 (3) Parallel processes for small tasks (`scripts/vpar.sh`)

PushT2dHard seeds 6–17, GN-in-cycles flags, wall time including process start-up (`results/vcycle/par.jsonl`):

| | Wall | Solved |
|---|---|---|
| one GPU process (as `vplf.sh`) | 110 s | 12/12 |
| 3 concurrent GPU processes (4 seeds each) | **66.5 s (1.65×)** | 12/12 |
| 4 concurrent CPU-only processes (`JAX_PLATFORMS=cpu`, 3 seeds each; after one warm-up process) | **68.9 s (1.6×)** | 12/12 |

The small tasks are launch-bound (the T's IP iteration costs the same with and without the dynamics Hessian), so the GPU has room for several processes and the CPU devices run the whole solver at ~2.2× the per-iteration cost (28 vs 12.8 ms). `vpar.sh <Task> <a:b> <label> <out> <k> [flags]` splits the seed range over k concurrent processes (`CPU=1` for CPU-only) and appends the rows per part. Not for the Allegro tasks (GPU memory; ≤ 3 Allegro jobs, §10.9) and not for timing measurements (per-seed times are contended).

### 11.8 Final evaluation of V14 (`results/vcycle/final.jsonl`, label V14; sequential, otherwise idle GPU)

| Task | Solved | Median s/seed | LPEC rounds |
|---|---|---|---|
| PushT2dHard 0–29 | 29/30 | 7.1 | seed 0: 1 round (3 biactive, 3 pressed) → 0.0045, 52 s; seed 4: 4 rounds (0.137 best; 3 rejections → stop), 170 s |
| PushT2d 0–17 | 18/18 | 2.9 | seed 11: 1 round (167 biactive, 1 pressed) → 0.0005 |
| PushT2dHardLong 0–17 | 17/18 | 6.7 | seed 0: 0.609, no biactive index (B-stationary, stop); seed 11: 1 round → 0.0010 |
| AllegroPen ×5 | 5/5 | 99.6 (70–108 s without LPEC) | V14p2: PLF-V 0.0745, LPEC Δ 0.05 and 0.02 rejected, Δ 0.008 → 0.0084, 738 s |
| PushSphere2d 0–9 via `vpar.sh` k=3 | 10/10 | – | – |

Pen plans: clearance ≥ margin in all 5 (`eval_hard.py`); position error 0.2–4.6 mm; orientation error 20–34° except V14p2 (133°, flipped; see §11.1).

Every row of the T family missed by V14 had been solved by some other run (PushT2dHard 4 by the GN runs, PushT2dHardLong 0 by C10 and GC1): the remaining misses are run-to-run, not structural.

### 11.9 Files, pitfalls, next

- `scripts/lfddp_v.py`: new flags `--lpec R`, `--lpec-delta`, `--ip-hess`, `--ip-hess-cycle`, `--ro-overlap`, `--v-local`; row fields `lpec` (rounds) and timing key `ro`. Pre-edit copy `scripts/lfddp_v_pre1123.py`.
- `scripts/lpec_step.py` (in-process LPEC), `scripts/hs_export.py` (export of the resumable IP loop), `scripts/vpar.sh` (seed range over k processes; `vplf.sh` takes `LOGSUF`), `scripts/ip_bench.py` (per-iteration IP cost, `--profile`), `scripts/ro_bench.py` (read-out pieces), `scripts/trace_sum.py` (GPU trace summary), chains `scripts/chain_{gn,b,c,d,e,f,g,h,i}.sh`.
- Rows: `results/vcycle/speed.jsonl` (GN/GC/EX/overlap/LPEC tests), `gate_gn.jsonl`, `gate_gc.jsonl`, `seeds.jsonl` (new seeds, EXn/GNn/GCn), `lpec_t.jsonl` (LP14, CTRL), `par.jsonl`, `final.jsonl`, `local.jsonl`; logs `results/vcycle/logs/`.
- Pitfalls: `pgrep -f <pattern>` in a wait loop matches the waiting shell itself when its command line contains the pattern (it deadlocked one chain; use `chain_x[.]sh`-style patterns inside scripts, never on the command line of the waiter), and `pkill -f` with such a pattern kills the caller (again, exit 144). Do not edit a shell script that a running chain is executing (bash reads it incrementally). Repeat runs are now close to deterministic on the T (GN1–3 identical), so repeats of the same seeds add little; use new seeds.
- Next: per-stage smoothing inside the lifted IP (§4.6 step 2, package edit) is the only form of "coarse on idle stretches" not yet tried; the pen's remaining IP cost is mostly the linearization (§11.4 correction); formerly stated as the sequential per-stage Riccati solve (§11.4, package KKT); the LPEC rescued plan can flip the pen (orientation is not in the metric).

## 12. Solve-time options tried (2026-11-23, afternoon)

Asked: try (1) CPU line-search threading, (2) per-process start-up, (3) parallel processes, (4) skipping redundant fine-level cycles, (5) a shared budget for LPEC rounds, (7) the cost of the pen's IP iteration, in order of fruitfulness. Where V14's time goes (`final.jsonl`): on PushT2dHard 7 % in instances without a cycle, 41 % in instances with cycles, 52 % in the two instances that reached the LPEC; phases: IP 36–47 %, LFDDP line search 22–31 %, read-outs 12–14 %, linearization 12–14 %.

| Option | Result | Verdict |
|---|---|---|
| (7) Derivative of the mass-matrix solve | The pen's batched triangular solves (16 ms of the 55 ms GN IP iteration) were forward-mode Jacobians differentiating the Cholesky factorization of the 22×22 Allegro mass matrix (in `forward.implicit` and the SAP context), with up to 1782 right-hand sides. `mjx/_src/small_la.spd_solve` (custom derivative `M⁻¹(db − dM x)`, `M⁻¹` formed once and applied as a product) in `forward.implicit`, and a gradient-stopped factor in `solver_sap.Context.create` (used only by the stop-gradient Delassus estimates and the initial guess, which `_solve`'s custom derivative ignores). n ≤ 16 keeps the old path. Jacobians equal to 5e-13 (pen), bit-identical (T). Pen IP iteration 54.7 → **45.9 ms** (GN), 142 → 129 ms (base); LFDDP linearization 39 → **28 ms**; JarC unchanged. V14 with the patch: pen 3/3, AllegroBallTableHard 10/10, Jar 5/5, JarC 5/5 (clearance 2/5, same seeds as before). | **adopted (package edit)** |
| (7) Same rewrite of the SAP rule `cho_solve(H, −df)` | no gain (48.6 vs 45.9 ms, noise) | reverted |
| (2) Export of the driver's jitted functions (`hs_export.ExportedJit`; LFDDP lin/bwd/fwd/rollouts, read-outs; on by default, `PLF_FN_EXPORT=0` disables) | same results (PushT2d seed 1 identical); process 30.7 → 27.0–28.6 s (PushT2d 1:3). Remaining start-up per process ≈ 27 s: imports ~7 s, loading the cached IP-loop executable ~9 s (a cache hit), model build ~3 s | adopted (small) |
| (1) Single-threaded Eigen on the CPU devices (`CPU1T=1` in `vplf.sh`) | microbenchmark: pen line search 45.7 → 31.4 ms per trial, T unchanged; full pen runs: 74–96 vs 87–91 ms per LFDDP iteration (≤ 3–5 % overall); roundoff-level path changes on Allegro and a cold recompile | not adopted (opt-in kept) |
| (1') The current par line search | already concurrent: 5 rollouts on 5 CPU devices 45.7 ms vs one 33.8 ms (pen) | – |
| (4) Stop an excursion when its own plan stalls before any true-law response | replay on 546 logged cycle excursions: would save 15k of 24.6k non-responding IP iterations but cut 39 of 237 responding excursions before their response, including the pen excursions that solved (0.597 → 0.465, 0.096 → 0.011) | rejected |
| (4') Skip the contracted finer levels | finer-level cycles were productive in 32/237 cases, several leading to solves (PushT2dHard 23: 1.14 → 0.015 at σ0/10; 14, 22; pen continuations) | rejected |
| (5) Shared budget for LPEC rounds | the successful round is often a later, smaller-radius one (PushT2d 10 and Long 10: round 2; pen V14p2: round 3) | rejected |
| (3) Parallel processes (`vpar.sh`), PushT2dHard 6:18, current code | 1 GPU process 108 s; k = 2: 113 s (unlucky split), **k = 3: 60 s (1.8×)**, k = 4: 63 s, k = 6: 60 s; 3 GPU + 3 CPU-only: 76 s, 4 GPU + 2 CPU: 102 s (CPU-only processes compete with the GPU processes' CPU read-outs and line search); all 12/12 | use k = 3 GPU processes for small-task sweeps |

Notes: the package edit invalidated the export cache once (first process per task re-exports, ~1–2 min). Not tried: a long-lived worker process serving many runs (would remove the ~27 s per-process start-up for repeated runs); fused kernels for the remaining linearization (command buffers 26 ms of the 46 ms pen iteration).
