# Handoff: PLF-V as a ladder search — canonical form, simplifications, and test plan

Date: 2026-11-23 (evening). Repo state: `master` at `44ca93a` plus the uncommitted package opt-ins listed by `git status`. Nothing in this plan has been implemented. The only new file besides this one is `scripts/exc_stats.py` (reads stored rows; no solver run).

Read first: `HANDOFF_fidelity_cycles.md` §10–§12 in this folder (PLF-V, V14, and the solve-time options already tried), and `../writeup_2026-11-23/plfv.tex` (Algorithms 1 and 2 are what this plan simplifies).

## 0. What is being asked, and the short version

**Question (user, 2026-11-23).** PLF-V (V14) performs well, but its description reads as heuristic: hardcoded reduction factors, two algorithms with many special cases. Is there a simpler framing, is PLF-V an instance of a broader automatic method, and would that method be better in success rate, convergence, or speed?

**Answer so far (analysis only; nothing run except statistics on stored rows).**

- V14 is one backtracking schema applied twice. Level 1 backtracks over the smoothing levels; level 2 backtracks over the LPEC radius and uses a whole level-1 search as its trial (§2). Algorithms 1 and 2 hide this because the same three tests are written out under different names at each call site.
- The two nested loops are structural. A single trust radius cannot replace them, because stationarity at a fine level says nothing about a coarse level.
- What is not canonical is a list of eleven exceptions (§2.3): per-call-site read-out and curvature choices, the terminal boosts, the step-size table, the per-task settings, and several separate budgets.
- I expect no gain in success rate from the canonical form. The gains to test are fewer rules, a termination statement stronger than monotonicity, and time in the LPEC rounds.
- Correction to my chat summary of the same day: I proposed stopping excursions early when the plan goal does not fall. `HANDOFF_fidelity_cycles.md` §12 item (4) had already replayed that rule and rejected it. This plan does not include it (§1.3).

**What this plan does.**

1. Rewrites the driver as the schema of §2 with a compatibility setting that reproduces V14 exactly (E1). Everything else is measured against that.
2. Removes the exceptions one at a time, each behind a flag, each against the gates of §4.9 (E2–E6).
3. Tests whether "stopped at a ladder-stationary point" is a real property of the returned plan (E7), which decides what the write-up may claim.
4. Tries one solver for both phases as a stretch (E8).
5. Gives a fallback for each step (§6).

For most steps the success criterion is **non-inferiority with fewer rules**, not a higher solve rate. Only H4, H6 and H8 claim speed.

**Terms.** As in `HANDOFF_fidelity_cycles.md` §0, plus:

| Term | Meaning |
|---|---|
| Rung | One model on an ordered ladder: a smoothing level (level 1), or an LPEC trust radius (level 2). |
| Trial | One attempt to improve the incumbent with one rung. Level 1: excursion, then continuation. Level 2: LPEC step, then a whole level-1 search. |
| Sufficient decrease | The trial's result has true goal ≤ (1 − τ) times the incumbent's, τ = 0.05. |
| Continuation | The feasible phase walking from the trial's level toward the fine level (sharpen or tighten the terminal term at each model-change event). |
| Watchdog | Steps that need not improve the true goal are allowed for a bounded number of iterations (the cap of 80 before the first true-law response). |
| Ladder-stationary | No rung of either level gives sufficient decrease from the incumbent. |
| Compat | The setting of the new driver that reproduces V14. |

## 1. Evidence this plan is built on

### 1.1 The same primitive under several names (read from `scripts/lfddp_v.py`)

| Primitive | Where it appears in V14 | Constant |
|---|---|---|
| Sufficient decrease of the true goal | cycle productivity (`jc` update); LPEC round acceptance; progress stall (`--prog-stall`); the progress window inside `explore` | τ = 0.05 |
| Model error against remaining distance | accuracy test at every model-change event; hand-off decision after an exploration; `auto` start (`restart_point`, `cycle_start`) | κ = 0.25 |
| Plan improves, truth does not follow | ratio test in `explore` and in the feasible phase | τ, η = 0.25, dwell 10 |
| Patience | stall 20; `--ip-stall` 20; cap 80 (`--warm-ip`); `--ip-cycle` 40; dwell 10 | five counts |
| Contract after failure, reset after success | cycle level `sig0 * 10**-jc`; LPEC radius `/2.5` with three strikes | 10; 2.5 |
| Restart a solve from the incumbent under a changed model | sharpen; boost; PLF's fine cycle (`--cycle 2`); fidelity cycle (`--vcycle 10`); LPEC round (`--lpec 6`) | five mechanisms |
| Newton step on the lifted linearization | lifted IP (gaps carried); LFDDP (gaps contracted by the rollout) | two solvers |

### 1.2 Measured on stored rows

Command (run from this folder; about 5 s):

```bash
PY=/home/gchou/PycharmProjects/mujodico-main-dev/.venv/bin/python
$PY scripts/exc_stats.py results/vcycle/final.jsonl results/vcycle/seeds.jsonl --task PushT2d
$PY scripts/exc_stats.py results/vcycle/final.jsonl results/vcycle/plfv_pen.jsonl results/vcycle/speed*.jsonl --task AllegroPen
```

T-pushing: 99 rows in which a fidelity cycle fired and no LPEC round ran. Labels V14, EXn, GCn, GNn, so curvature variants are mixed. Single runs.

| Quantity | Value |
|---|---|
| Coarse excursions | 261: 111 productive (4,090 IP iterations), 150 unproductive (12,000; every one ran to the cap of 80) |
| By level, productive / unproductive | σ0: 101 / 94; σ0/10: 8 / 26; σ0/100: 2 / 17; σ0/1000: 0 / 13 |
| Incumbent below 0.03 at the start | 5 productive, 72 unproductive |
| First true-law response in productive excursions | median 15 iterations; 80 % by 30; 94 % by 60 |
| True-goal decrease in the feasible phase after a cycle (sum of log decrease per read-out) | σ = 0.1, b = 0: 177; all phases with b ≥ 1: 215; σ = 0.001 in total: 51; the productive excursions themselves: 104 |
| Accepted LFDDP step size at read-outs | 1: 24 %; 0.5: 19 %; 0.25: 17 %; 0.1: 19 %; 0.03: 21 % |
| Time split (1,376 s) | IP 510; line search 369; linearization 175; true rollouts 175; backward pass 82 |

Readings:

- Most of the descent after an excursion happens in the feasible phase at σ = 0.1, and about half of it in phases with a terminal boost. The excursion is a kick; the continuation does the work. I had expected the ladder to over-polish between strokes, and the rows do not support that. Whatever replaces the boosts must deliver the same decrease.
- Only one accepted step in four is a full step, and one in five is at the smallest step size of the grid. The step control of the feasible phase is the least settled part (H4).

LPEC rounds stored in `final.jsonl` and `lpec_t.jsonl` (11 rows):

- A rejected round costs 23–45 s on the T and 125–203 s on the pen. PushT2dHard seed 4: 133 of 170 s in four rounds. `HANDOFF_fidelity_cycles.md` §12: 52 % of V14's PushT2dHard time is in the two instances that reached the LPEC.
- Of the 10 accepted rounds, 6 needed no fidelity cycle inside the round, and the others 3, 4, 4 and 10 (the last one unsolved).
- A row stores the cycle log of its best round only. What rejected rounds spend their time on is not logged (first todo of E6).

### 1.3 Already closed: do not redo

| Idea | Result | Where |
|---|---|---|
| Stop an excursion when its own plan stalls before any true-law response | Rejected: cuts 39 of 237 responding excursions, among them the pen excursions that solved. `exc_stats.py` agrees: a plan-goal gate at iteration 40 would cut 2 of 95 τ-productive excursions on the T but 12 of 68 on the pen | fidelity §12 (4) |
| Skip the contracted finer levels | Rejected: productive in 32 of 237 cases, several leading to solves | fidelity §12 (4') |
| A shared budget across LPEC rounds | Rejected: the successful round is often a later one with a smaller radius | fidelity §12 (5) |
| Level chosen by the accuracy test | Negative: far incumbents get fine levels and creep | fidelity §10.8 |
| Closed-loop read-out in PLF's first exploration | Negative: pen 0/2 | fidelity §10.3 (PU) |
| Gauss–Newton in every IP call | Negative: PushBox2d seed 5, pen 4/9 | fidelity §11.5 |
| Coarse only on idle stretches, with LFDDP on a mixed model | Negative: no new face discovered | fidelity §11.3 |
| Automatic switch between simulated and true restart points in base PLF | Negative: fingers-hard 15/24, PushBox2d 4/6 | trust-ratio §14.2 |
| A 3-point step grid | Negative: more iterations | trust-ratio §14.3 |
| Smoothing tied to the barrier; optional priced forces | Negative | `HANDOFF.md` in this folder; rows in `../handoff_2026-11-14/followup/results/pricing_2026-11-21/` |
| True-residual LMS as one solve | Negative | trust-ratio §1.4 |
| Windows, subgoals, tailored discovery phases, sampling of contact sequences | Excluded by the user for now | — |

## 2. The canonical form

### 2.1 One schema

```
Search(U, rungs, trial, budget):
    k ← 0
    while r(U) > tol and k < len(rungs) and budget remains:
        U' ← trial(U, rungs[k])                 # never worse than U
        k  ← 0 if r(U') ≤ (1 − τ) r(U) else k + 1
        U  ← U'
    return U
```

### 2.2 V14 as two instances of it

```
Level 1   rungs = (σ0, σ0/10, σ0/100, σ0/1000)        budget = V cycles
          trial(U, σ) = Continue(Excursion(U, σ), σ)

Level 2   rungs = (Δ0, Δ0/2.5, Δ0/2.5²)                budget = R rounds
          trial(U, Δ) = best(U, Level1(U + LPEC(U, Δ)))  ;  B(U) = ∅ → stop with a certificate

Run       U ← Continue(Excursion(0, σ0), σ0)  ;  U ← Level1(U)  ;  U ← Level2(U)
```

- `Excursion(U, σ)`: lifted Newton steps at σ from the lift of U (its simulation at σ if that reproduces r(U) within κ, else its true states with forces of the level-σ law). True read-out every 5 iterations. It ends on the ratio test, on 20 iterations without sufficient decrease, or at the watchdog (80 iterations without any true response).
- `Continue(U, σ)`: the feasible phase from U at σ. At each model-change event: sharpen if the accuracy test fails, else tighten the terminal term. At the fine level with the terminal term exhausted: return.
- Level 1 on success returns to the top rung. Level 2 on success keeps its radius and resets only its strike count (difference D1, tested in E6).

**Finite termination without a cycle budget.** Each success multiplies the true goal by at most 1 − τ, so a search has at most about ln(r₀/tol) / (−ln(1 − τ)) successes (about 111 for r₀ = 3, tol = 0.01, τ = 0.05), each followed by at most len(rungs) failures. V and R are then caps on cost and not part of the method. This can be stated in the write-up without any experiment.

### 2.3 Where V14 is not canonical (the exceptions)

| | Exception | Status |
|---|---|---|
| X1 | Read-out: open loop in the first exploration and in PLF's fine cycles; TVLQR closed loop in fidelity excursions; best of open loop and DDP gains in the feasible phase | closed loop in the first exploration is negative; stays an argument of `Excursion` |
| X2 | Curvature: dynamics curvature in base IP calls; Gauss–Newton in fidelity excursions | Gauss–Newton everywhere is negative; stays an argument |
| X3 | Stall rule: "any new incumbent" before the first fidelity cycle, sufficient decrease after it; watchdog only in fidelity excursions | **H2** |
| X4 | PLF's two fine cycles: an exploration at the current model that is not a rung | **H2** (as the bottom rung of level 1) |
| X5 | Terminal boosts ×10, at most two | **H3** |
| X6 | Step grid, step-size-to-regularization table, ×10 on failure, merit weight 10³ | **H4** |
| X7 | Per-task block of `vplf.sh`: barrier values, σ0 = 10 on the knob, restart point | **H5** (barrier and σ0 only; the restart point stays, §1.3) |
| X8 | Level 2 wraps a whole level-1 search; factor 2.5 against 10; D1 | **H6** |
| X9 | Budgets: 250, 400, 400 per cycle, V = 10, R = 6, caps 80 and 40 | reduced by H2 and H6; the rest are cost caps |
| X10 | Barrier schedule: reset per cycle, ×0.1 per sharpening in the feasible phase, fixed in the IP | not addressed |
| X11 | Two solvers | **H8** (stretch) |

## 3. Hypotheses

Thresholds are first choices, not tuned values. "Gate" and "hard set" are defined in §4.9.

| | Hypothesis | Prediction | Falsified if | Tested in |
|---|---|---|---|---|
| H1 | The schema of §2.2 is V14 | Under deterministic ops the new driver in compat mode gives rows identical to `lfddp_v.py` (events, iteration counts, goals to 1e-9) on the probe set of E1 | Not a scientific hypothesis: any difference is a bug to find before going on | E1 |
| H2 | One stall rule and one watchdog serve every call site, and the fine cycles are the bottom rung | Gate within ±1 seed per task in both runs, fingers-hard ≥ 23/24; hard set within ±2 of V14; medians within +10 % | A gate task loses ≥ 2 seeds in both runs, or the pen loses ≥ 2 of 5 | E2 |
| H3 | A multiplier on the terminal error replaces the boosts | Same solve-rate criteria as H2; iterations in the feasible phase per solved instance not higher; no multiplier component changes sign on three consecutive updates | Loses ≥ 2 seeds on a task in both runs, or oscillates on ≥ 3 instances | E4 |
| H4 | A gain-ratio rule replaces the step-size-to-regularization table | Full steps rise from 24 % to ≥ 35 % of accepted steps on the T rows; fingers-hard median iterations ≤ the current value; line-search time per instance not higher | Iterations rise by > 10 % in both runs on fingers-hard or the T | E3 |
| H5 | The top rung can be discovered, which removes the barrier and σ0 entries of the per-task block | One start for all tasks: PushSphere2d ≥ 9/10, PushBox2d ≥ 5/6, knob ≥ 18/20, fingers-hard ≥ 23/24, PushT2dHard within 2 of V14; medians within +10 % on fingers and +25 % on pushers | Any task drops by ≥ 2 seeds in both runs | E5 |
| H6 | An LPEC round shows early whether it will be accepted | In logged rounds, accepted ones pass below (1 − τ) of the outer incumbent within c\* cycles (stored accepted rounds: 0 cycles in 6 of 10); a cap of c\* + 1 cycles in a round loses no accepted round in replay and saves ≥ 30 % of the time of LPEC instances | The cap cuts ≥ 2 accepted rounds of the logged set | E6 |
| H7 | "Ladder-stationary" is a property of the returned plan | Under deterministic ops, restarting from a plan returned as ladder-stationary returns the same plan in ≥ 90 % of cases, and every miss falls into one of three stop classes (§4.7) | < 50 % are fixed points (then only monotonicity may be claimed) | E7 |
| H8 | The feasible solver with carried gaps explores as well as the lifted IP | From stored cycle-start incumbents, the productive fraction of σ0 excursions is within 10 points of the IP's on the same incumbents, and the pen iteration is cheaper | Productive fraction lower by > 20 points on the T, or no pen slide escaped in 3 tries | E8 |

## 4. Methods

Conventions as in `HANDOFF_fidelity_cycles.md` §4. All new code is script-level in `scripts/`; the package export cache (`~/.cache/mujodico_ippas/export`) is untouched until §4.3's optional second phase.

### 4.1 The driver `scripts/lfddp_l.py`

- Start from a copy of `lfddp_v.py`. Do not edit `lfddp_v.py` or `lfddp.py`: other sessions' scripts launch them, and they are the reference for H1.
- The `LFDDP` class and the closures `restart_point`, `start_point`, `cycle_start`, `readout_ip`, `explore` stay as they are. Only the control flow from "phase B" to the end of the round loop is rewritten as:
  - `excursion(U, rung, mode)`: a thin wrapper of `explore` whose arguments are the X1–X3 choices (read-out, curvature, stall rule, watchdog, cap).
  - `continue_(X, U, C, sig, state)`: the feasible-phase loop up to the point where it has no model change left, returning the incumbent and a stop reason.
  - `search(U, rungs, trial, budget)`: §2.1, used for both levels.
- `--compat v14` (default) sets every argument so that the path is V14's. Each later change is one flag that departs from compat.
- Row format unchanged, plus: `stop_class` (§4.7), one `cyc_log` per LPEC round (`rounds` field), and the cycle-start incumbents when `--save-cyc 1` (`SAVE_DIR/<label>_<Task>_<seed>_c<n>.npz`).
- Launcher: copy `vplf.sh` to `lplf.sh` with `SCRIPT=lfddp_l.py` and rows in `results/ladder/`. Gate: `vgate.sh` with the launcher swapped.

Subtleties to carry over exactly (each one changed the path when it was introduced):

- `v_reset always`: the ladder restarts at the cycle level after every excursion, productive or not; productivity is judged on excursion plus continuation at the next trigger.
- The budget arithmetic `budget + v_budget * n_v` and `total_budget + v_budget * (n_v + 1)`; `force_cycle` on a budget end.
- `last_acc = it - stall` after an unproductive cycle (forces the next exhaustion check at once).
- The first exploration of a `--from-us` or LPEC round is a cycle-style excursion at `cycle_level(0)`.
- The IP is compiled with `max_iter = total_budget` when `v_guard >= 3`.

### 4.2 One stall rule, one watchdog, fine cycles as a rung (H2)

Flag `--rules one`:

- `prog_stall = 2`: sufficient decrease is the stall test from the start, in the feasible phase and in every excursion.
- `stall_guard = True` in the first exploration and in PLF's fine cycles: before the first true response only the cap ends the excursion. This is the "untested" item of fidelity §10.10.
- PLF's fine cycles become the last rung of level 1: an excursion at σmin, with the count 2 replaced by the schema's own rule (one trial; repeated only after a success). `--cycle 0` and rungs `(σ0, …, σmin, σmin*)`, where the starred rung keeps the fine cycle's arguments (cap 40, open-loop read-out, base curvature, current terminal term).
- X1 and X2 are not changed here.

Test the three items separately first (E2); adopt each that passes.

### 4.3 Terminal multiplier (H3)

- Terminal cost today: `eᵀ Qf e · 10^b`, quaternion blocks with the distance `d = 1 − ⟨q, q_g⟩²`.
- Replacement, flag `--term mult`: `eᵀ Qf e + νᵀ e + Σ_blocks ν_q d_q`. At an event where V14 would boost: `ν ← ν + 2 Qf e` and `ν_q ← ν_q + Qf_q d_q`, with e and d_q the terminal error of the feasible-phase iterate. The plan's own error is used on purpose: a stalled, accurate plan is the case where the penalized optimum itself misses (trust-ratio §4.4).
- The "two boosts used" condition becomes: the last update did not give sufficient decrease of r\*. Then sharpen; at σmin, return.
- Phase 1, script-level: `lterm` is defined in the driver's `LFDDP` class, so the linear term goes there (ν passed with `P`). Excursions already run with the base terminal weight. The fine-cycle IP runs with the base weight and no multiplier in this phase; that is an approximation to state in the results.
- Phase 2, only if phase 1 passes: a `nu_f` field in `util.TranscriptionParams` and the linear term in the transcription's terminal cost, so the IP sees the same model. This is a package edit: keep a copy of the export cache, and check that zero `nu_f` is bit-identical under deterministic ops before any run.
- Do not implement the multiplier as a shift of `x_goal`: the stage cost reads the same `x_goal`, so the shift would also move the running-cost target.

### 4.4 Gain-ratio regularization (H4)

- Today: after an accepted step the regularization is multiplied by a factor looked up from the step size (1 → 0.25, 0.5 → 1, 0.25 → 3, 0.1 → 10, 0.03 → 30); on a failed line search or Cholesky ×10.
- Replacement, flag `--lm 2`: with `ρ = (J − J_new) / (−(α d1 + ½ α² d2))` for the accepted step (both already computed; with gaps present use the merit values of the acceptance test), `reg ← reg · max(1/3, 1 − (2ρ − 1)³)`; on failure `reg ← reg · ν_r`, `ν_r ← 2 ν_r`, reset `ν_r = 2` after an acceptance. Bounds 1e-9 and 1e6 as now.
- The parallel step grid stays (five concurrent CPU rollouts cost about one; the 3-point grid was worse).
- Log per accepted step: α, ρ, reg. `exc_stats.py` already reports the α distribution.

### 4.5 A discovered top rung (H5)

- Two barrier sets, the existing class values: `B_lo = (ip-mu 1e-3, ip-mumin 1e-2, mu0 1e-3)` and `B_hi = (0.1, 0.1, 1e-2)`.
- Flag `--ladder auto`. First exploration, ascending until the true law responds: `(σ 1, B_lo)`, `(1, B_hi)`, `(10, B_hi)`. "Responds" = a read-out improves on r(0) by τ. This is PIP4F's widen order (barrier first, then smoothing), which gave 89/90 on the gate and PushBox2d 6/6 (trust-ratio §14).
- The rung that responded becomes the top rung of level 1 for the rest of the run. If level 1 is later exhausted, try one rung above the top once before level 2.
- `--handoff-from` and `--restart` keep their per-task values (the measured automatic switch was negative).
- Cost on tasks that today start high: up to two failed excursions of at most 80 iterations. That is why H5 allows +25 % on pushers.

### 4.6 LPEC rounds (H6)

1. Logging first: `rounds` field with each round's `cyc_log`, events and the running minimum of its local incumbent per cycle.
2. Replay on logged rounds: for a cap c on the cycles inside a round, count accepted rounds lost and seconds saved.
3. If the replay passes: flag `--lpec-cyc c`.
4. D1, separately: flag `--lpec-reset 1` returns the radius to Δ0 after an accepted round (as level 1 does). One run on the stuck-plan set of E6.
5. Not to be tried again: fewer rounds or a shared budget across rounds (§1.3).

### 4.7 Stop classes and the fixed-point test (H7)

Every row gets `stop_class`:

| Class | Condition |
|---|---|
| `goal` | r\* ≤ stop tolerance |
| `budget` | a cost cap ended a search whose last trial gave sufficient decrease, or level 2 ended at R |
| `stationary-B0` | level 1 exhausted and the LPEC finds no biactive index |
| `stationary` | level 1 exhausted and level 2 exhausted (three strikes) with a non-empty biactive set |

Fixed-point test: under deterministic ops, restart the driver from each plan of the two stationary classes (`--from-us`) with the same flags; record whether the returned plan is the same (true goal within 1 % and controls within 1e-6). Restart the `budget` plans too: they should improve, and if they do not they are misclassified.

### 4.8 One solver (H8, stretch)

- A second forward pass in the `LFDDP` class, `forward_lifted`: `δx` from the linear rollout of the Riccati policy on `(A, B, a)`, `δu = k + K δx`, `δγ = Φ_x δx + Φ_u δu − h̄`, then `x̂ = x + α δx`, `û = clip(u + α δu)`, `γ̂ = γ + α δγ`. Gaps are not closed by a rollout; the next linearization recomputes them.
- Acceptance on the existing merit (cost plus `nu_merit` times the 1-norm of the gaps), with a fraction-to-boundary rule on the normal forces so the barrier stays defined.
- The barrier then acts on the lifted force itself. On inconsistent iterates that is the pull that grows as the smoothing shrinks (trust-ratio §12, the ProxDDP note), which is the suspected discovery ingredient of the IP.
- Used only as the engine of `excursion` (`--exc-engine lifted`); the continuation keeps the rollout pass.
- Known risks: sliding contacts carry exactly zero force on some pyramid edges (the barrier is on the normal force for that reason; individual edges need a projection to ≥ 0); the IP's inequality handling and filter are absent.

### 4.9 Gates

| Gate | Definition |
|---|---|
| Solve rate | Gate set (90) and knob (20), two runs with default ops. Adopt only if no task loses ≥ 2 seeds against V14 in both runs and fingers-hard is ≥ 23/24 in both |
| Hard set | PushT2dHard 0:30 (V14: 29), PushT2d 0:18 (18), PushT2dHardLong 0:18 (17), AllegroPen ×5 (5), AllegroJar and JarC 0:5 (5, 5). One run, plus one run on new seeds (PushT2dHard 30:54; check first that these seeds give valid instances). Repeats of the same T seeds add little (fidelity §11.9) |
| Speed | Median seconds per seed within +10 % of V14 per task; idle GPU, default ops, first seed of a process excluded |
| Equivalence | Deterministic ops (`DET=1`), rows compared field by field |

V13 against V14 differed by 1–2 instances out of 24–30 on the T with different seeds failing. A difference of that size is not evidence in either direction.

## 5. Experiments

**Order of work.** E0 → E1, then E3 (independent of the rest), E2, E4, E6, E7, E5, E8. E9 closes every adopted change. If only one day is available: E1, E3, and the write-up rewrite of todo T0.

### E0. Logging and statistics (two hours)

- `exc_stats.py` exists and reproduces §1.2.
- Add to the copy: per-round logs, cycle-start incumbents, per-step α/ρ/reg, `stop_class`.

### E1. Equivalence (half a day to one day; H1)

- Probe set under `DET=1`: PushT2dHard 0:6, PushBox2d 0:6, fingers-hard 0:6, PushSquareKnobTable 8:12 (contains seed 10, where a cycle fires), AllegroPen once, and one LPEC case (`--from-us results/vcycle/plans/GCn_PushT2dHard_{seed}.npz` on seed 25, with `--lpec 6`).
- Run `lfddp_v.py` and `lfddp_l.py --compat v14` back to back; diff `events`, `cyc_log`, `lpec`, `it`, `ip_its`, `true_goal`.
- Pass: identical. Then one timing pair on PushBox2d and fingers-hard with default ops (the per-stage variant once changed the path only by compiling differently; fidelity §10.7).

### E2. One rule set (half a day; H2)

Arms on fingers-hard 0:24, PushBox2d 0:6, PushT2dHard 0:12, AllegroPen ×3, one run each, then the adopted combination through E9:

| Arm | Change from compat |
|---|---|
| a | sufficient-decrease stall from the start |
| b | watchdog in the first exploration and the fine cycles |
| c | fine cycles as the bottom rung |
| abc | all three |

Report per arm: solved, iterations, and the number of excursions that ended at the watchdog without a response.

### E3. Gain-ratio regularization (half a day; H4)

- Fingers-hard 0:24 and PushT2dHard 0:12, two runs each, `--lm 1` (compat) against `--lm 2`.
- Report: solved, median iterations, α distribution, median and maximum reg, line-search seconds per instance.
- Pen once per arm: the feasible phase crept at reg 1e4–1e5 for 350 iterations before the sufficient-decrease stall existed (fidelity §10.3, PV8); report the maximum reg.

### E4. Terminal multiplier (one day; H3)

- Phase 1 arms on fingers-hard 0:24, PushT2dHard 0:12, knob 0:20, PushBox2d 0:6, pen ×3: compat; `--term mult`; and the control with neither boost nor multiplier (`--boosts 0`).
- Report: solved, iterations in the feasible phase, the true-goal decrease booked to terminal-tightening phases (the `exc_stats.py` table), sign changes of ν.
- Phase 2 only if phase 1 passes on every task.

### E5. Discovered top rung (one day; H5)

- `--ladder auto` with no per-task barrier or σ0 flags on PushSphere2d, PushSphere2dHard, PushMultiSphere2d, PushBox2d, knob, fingers, fingers-hard, AllegroBallTableHard, PushT2dHard 0:12; two runs.
- Report per task: solved, the rung that responded on each seed, iterations before the response, median time against V14.

### E6. LPEC rounds (half a day of logging and replay, then one day; H6)

- Stuck-plan set, all present in `results/vcycle/plans/`: the six T misses `GCn_PushT2dHard_{23,25}`, `GCn_PushT2d_{10,11}`, `GCn_PushT2dHardLong_{10,11}` (the starts of the CTRL rows), and the pen slides `PV11c`, `PV11d`, `PV2` (`_AllegroPen_0.npz`). The `V14_*` files are final plans after their rounds, not stuck starts; for V14's PushT2dHard seeds 0 and 4 rerun from scratch with logging.
- Run each with per-round logging (at most three pen jobs at a time).
- Replay caps c = 0, 1, 2, 4. If a cap passes H6: rerun the set with `--lpec-cyc`, then D1.

### E7. Stop classes and fixed points (half a day; H7)

- `DET=1`, PushT2dHard 0:30 and PushT2dHardLong 0:18 with the current best flags; then the fixed-point test of §4.7 on every miss, and on five solved plans as a sanity check (they must return immediately).
- Report the class counts and the fixed-point fraction per class.

### E8. One solver (two days; H8; only after E1–E4)

- Inputs: cycle-start incumbents saved in E1–E6 (at least 20 from the T with known outcomes at σ0, and the pen slides).
- One σ0 excursion from each with the IP and with `--exc-engine lifted`, same read-out and stop rules, deterministic ops.
- Report: productive fraction, gain, iterations to the first response, milliseconds per iteration (T and pen).
- Full runs only if the productive fraction passes.

### E9. Gates (one day per adopted change)

§4.9, all four. Then update `results/ladder/L<n>flags.txt` and the summary table.

## 6. Fallbacks

**F1. E1: the new driver cannot be made identical.**

1. Bisect by event: print `events` from both drivers and stop at the first difference; the cause has always been a counter (`last_acc`, `last_change`, the budget arithmetic) or a different compilation of the same function.
2. If the difference is numerical only (same events, goals differ after ~1e-12): accept equivalence on events and counts, and replace the exact test by a two-run gate.
3. If a day is spent without convergence: keep `lfddp_v.py` as the implementation, add the flags of E2–E6 to a copy of it instead, and use §2 for the write-up only. The experiments do not depend on the refactor.

**F2. E2: a unified rule loses seeds.** Keep the arm that passed and record the failing one as a documented argument of `Excursion`. Expected failure mode: the watchdog in the first exploration lets a long discovery run on fingers-hard, where a long discovery phase at a high barrier causes the clamp (`../handoff_2026-11-14/followup/PRINCIPLES.md`, P4). In that case test the watchdog with a cap of 40 in the first exploration only before giving it up.

**F3. E3: the gain-ratio rule needs more iterations.**

1. Check ρ on the accepted small steps: if ρ ≈ 1 at α = 0.03, the model is fine and the step is limited by the control box or by gap closing, not by curvature; then regularization is the wrong lever and the table stays.
2. Try the ratio rule only when no gaps are present (pure cost decrease), the table otherwise.
3. Otherwise keep the table and state it in the write-up as a Levenberg–Marquardt rule driven by the accepted step size.

**F4. E4: the multiplier oscillates, or loses the decrease that boosted phases deliver.**

1. Damp the update (half step) and update only when the plan's terminal error fell since the last update.
2. Combine: multiplier update plus one ×10 of the weight when the update gives no sufficient decrease (the standard augmented-Lagrangian penalty rule). This still removes the cap of two.
3. Otherwise keep the boosts and describe them as penalty continuation with the trigger "accurate model, stalled short of the goal" (trust-ratio F7).

**F5. E5: one start does not cover every task.**

1. If only the knob fails: it responds at σ = 1 on about half the seeds and then never climbs. Use the rule "after level 1 is exhausted, try one rung above the top" before level 2, which §4.5 already has; if the gain comes too late, climb when the first response is weaker than 25 % of r(0) (PIP4F's widen threshold).
2. If fingers-hard fails: the barrier rung is being climbed too eagerly. Require two excursions without response before `B_hi`.
3. Otherwise keep the per-task values and state that the search unifies the cycles only (as fidelity F10).

**F6. E6: accepted rounds do not show early.** Keep the full nested search per round. The time of LPEC instances then stays as it is; say so. Do not shorten the number of rounds.

**F7. E7: stationary stops are not fixed points.**

1. Look at what the restart changed: if a different excursion responded, the stop was a watchdog or budget stop in disguise; move those cases to `budget` and re-count.
2. If < 50 % remain fixed points: the write-up claims monotonicity and finite termination (§2.2) only, and reports the stop classes as observed outcomes.

**F8. E8: the carried-gap pass does not discover.**

1. Add the lifted barrier on every pyramid edge with a fraction-to-boundary rule (closer to the IP).
2. Use the IP's linear step with the feasible solver's read-outs (same Newton system, IP globalization) to see whether the globalization or the barrier is the missing part.
3. Otherwise stop: two solvers stay, described as one linearization with two globalizations. This matches fidelity §11.3.

**F9. A gate fails after adoption (E9).** Revert the last flag in the flags file; rerun the gate once to confirm the earlier state; do not trade gate seeds for hard-task seeds.

**F10. Nothing beyond E1 is adopted.** The plan still delivers §2 and the termination bound for the write-up, with X1–X11 listed honestly as the method's arguments. That is a presentation gain only, and should be reported as such.

## 7. Pitfalls already paid for

- From fidelity §7, §10.9, §11.9, all still valid: judge on the true rollout only; deterministic ops for equivalence and never for timing; at most three Allegro jobs on the 16 GB GPU, and check for other sessions' jobs first; kill by PID (`pkill -f` and `pgrep -f` match the caller); do not edit a shell script that a running chain is executing; the first seed of a process pays start-up.
- A numerically equivalent rewrite can change the path through compilation alone (per-stage vectors, fidelity §10.7). E1 must be passed again after any change to a jitted function.
- The pen's metric weights orientation 0.01; report height and orientation error separately, and check clearance with `eval_hard.py`.
- A package edit invalidates the export cache once (1–2 min per task) and is confined to E4 phase 2.
- The retrospective numbers of `exc_stats.py` are filters on logged streams. Stopping or changing an excursion changes everything after it.

## 8. What is not known

- Whether the driver can be restructured with a bit-identical path (E1).
- Whether sufficient decrease as the only stall test is too strict early in a solve, where V14 accepts any new incumbent (E2).
- Why 76 % of accepted feasible-phase steps are damped: curvature, the control box, or gap closing (E3, F3 item 1).
- Whether a multiplier is stable when the inner solves are never converged (E4; open since trust-ratio §8).
- Whether a response at a low rung on the knob prevents the climb to the wide smoothing it needs (E5).
- What rejected LPEC rounds spend their time on (E6).
- Whether the same instance stops at the same plan in two deterministic runs from nearby starts, i.e. how path-dependent "ladder-stationary" is (E7). Every V14 miss on the T was solved by some other run.
- Whether X1 and X2 can ever be removed. Both have a measured negative result on the pen and none on the T.
- X10 (the barrier schedule) has no proposal here.

## 9. Todos

Write-up, no code:

- [ ] T0. Rewrite Algorithms 1 and 2 of `plfv.tex` as §2.1–§2.2, with X1–X11 as stated arguments, and add the finite-termination bound. Faithful to V14 as it is; needs no experiment.

Code and experiments:

- [ ] T1. `scripts/lfddp_l.py` (copy; `excursion`, `continue_`, `search`; `--compat v14`), `scripts/lplf.sh`, `results/ladder/`.
- [ ] T2. Logging: `rounds`, `--save-cyc`, per-step α/ρ/reg, `stop_class`.
- [ ] T3. E1 equivalence under `DET=1`; timing pair.
- [ ] T4. `--lm 2`; E3.
- [ ] T5. `--rules one` in three separable parts; E2.
- [ ] T6. `--term mult` phase 1; E4. Phase 2 (package) only on a pass.
- [ ] T7. E6: logged rounds on the stuck-plan set; replay; `--lpec-cyc`; `--lpec-reset`.
- [ ] T8. E7: stop classes and the fixed-point test.
- [ ] T9. `--ladder auto`; E5.
- [ ] T10. `--exc-engine lifted`; E8 (stretch).
- [ ] T11. E9 for each adopted flag; flags file; results section appended to this document (§10, in the format of fidelity §10–§12).

## 10. File map

| Path | What |
|---|---|
| `scripts/exc_stats.py` | Statistics of §1.2 from stored rows (new) |
| `scripts/lfddp_v.py` | V14 driver: the reference for H1; do not edit |
| `scripts/vplf.sh`, `scripts/vgate.sh`, `scripts/vpar.sh` | Launchers (per-task block = X7); gate; parallel seed ranges |
| `scripts/lpec_step.py` | In-process LPEC step (`make_lpec`, constants `GAP_TOL`, `F_TOL`, `M_F`, `M_G`) |
| `scripts/pip.py` | PIP4F; `--widen` is the model for §4.5 |
| `scripts/hs_export.py`, `scripts/ippas_resume.py` | Resumable, exported IP loop used by `explore` |
| `scripts/eval_hard.py`, `scripts/pen_orient.py`, `scripts/sumv.py` | Plan checks and summaries |
| `results/vcycle/V14flags.txt` | Flags of the reference |
| `results/vcycle/final.jsonl`, `seeds.jsonl`, `lpec_t.jsonl`, `plfv_pen.jsonl`, `speed*.jsonl` | Rows used in §1.2 |
| `results/vcycle/plans/` | Stored plans (stuck-plan set of E6) |
| `../../examples/methods/util.py`, `../../examples/methods/transcription_lifted_multiple_shooting.py` | `TranscriptionParams`, terminal cost: the call sites of §4.3 phase 2 |
| `../writeup_2026-11-23/plfv.tex` | Write-up; Algorithms 1 and 2 (T0) |
| To write | `scripts/lfddp_l.py`, `scripts/lplf.sh`, `results/ladder/` |

## 11. Results (2026-11-23, late evening)

Done: T0, the reference check behind H1, and E3. Not started: T1 (the `excursion` / `continue_` / `search` refactor, E1), E2, E4–E8. Rows in `results/ladder/` (`e3.jsonl`, `e3c.jsonl`; `e3b_rho_bug.jsonl` is invalid, see below), launcher `scripts/lplf.sh`, driver copy `scripts/lfddp_l.py`, statistics `scripts/e3_stats.py`.

### 11.1 T0: the write-up

`../writeup_2026-11-23/plfv.tex` now states the method as §2: Algorithm 1 is the ladder search (rungs, one trial per rung, sufficient decrease, stop after F consecutive failures, a reset rule after a success: top rung at level 1, the same radius at level 2), Algorithm 2 is PLF-V as the first exploration plus continuation followed by two searches, with the trials T1 and T2 and the phases `Excursion` and `Continue` defined in the box. Added: Proposition (finite termination; at most F(1 + ⌈ln(r0/η)/(−ln(1−τ))⌉) trials, 112 successful ones for r0 = 3), the four stop states, and a table of the settings that differ by call site (X1–X11, with what was tried). The old "Fidelity cycle" box is gone; its content is the trial T1. PDF rebuilt (17 pages, no undefined references).

### 11.2 The reference (H1 prerequisite)

`lfddp_v.py` was edited after this plan was written (another task of the same day: LPEC rounds as a function `run_round`, `--lpec-solver`, `--lpec-par`, `--lpec-par-lazy`, `--ls-spec`, counters; all off by default). Under `DET=1` the current file with default flags reproduces the committed V14 file (`44ca93a`, kept as `scripts/lfddp_v_44ca.py`) bit for bit on PushT2dHard 0–2 and PushSquareKnobTable 10 (a cycle fires): goals, iteration counts and events identical. E1 can therefore use either file as the reference.

**Pitfall found:** `opt()` in `lfddp_v.py` takes the **first** occurrence of a flag, and `vplf.sh` puts its BASE block (`--lm 1`, `--warm-ip 80`, …) before the extra flags, so an extra flag that repeats a BASE or class flag is silently ignored. The first E3 arm ran `--lm 1` twice for that reason. `lfddp_l.py` takes the last occurrence. No earlier experiment repeated a BASE flag.

### 11.3 E3: gain-ratio regularization (H4) — falsified

`--lm 2` in `lfddp_l.py`: after an accepted step `reg ← reg · max(1/3, 1 − (2ρ − 1)³)` with ρ the actual over the predicted decrease of the acceptance test's merit; on a failed attempt `reg ← max(reg ν_r, 1e-4)`, `ν_r ← 2 ν_r`, reset to 2 after an acceptance; `--log-steps 1` logs (α, ρ, reg) per accepted step. (A first corrected run, `e3b_rho_bug.jsonl`, computed ρ after the cost had been overwritten, so ρ = 0 and reg doubled every step; discard it.)

| Task | Arm | Runs | Solved | Median feasible-phase its | Median s/instance | Accepted α = 1 / 0.5 / 0.25 / 0.1 / 0.03 | Median reg |
|---|---|---|---|---|---|---|---|
| fingers-hard 0:24 | table (`--lm 1`) | 4 | 96/96 | 28–31 | 2.6–2.7 | 16–19 / 23–30 / 24–27 / 17–20 / 12–13 % | 27–36 |
| fingers-hard 0:24 | gain ratio | 2 | 48/48 | **68–80** | **7.2–8.0** | 4–5 / 3–4 / 7–8 / 24–28 / **55–63** % | 0.3–0.5 |
| PushT2dHard 0:12 | table | 4 | 48/48 | 122–199 | 5.6–7.8 | 24–27 / 15–24 / 14–17 / 15–19 / 20–26 % | 2e4–4e4 |
| PushT2dHard 0:12 | gain ratio | 2 | **10/12, 12/12** | 73–94 | 4.7–5.2 | 13–14 / 8–9 / 12 / 20–21 / 46 % | 3e2–9e2 |
| AllegroPen | table | 2 | 2/2 | 179–200 | 83–88 | 21–27 / 19–24 / 16–19 / 18–19 / 17–19 % | 15–30 |
| AllegroPen | gain ratio | 1 | 1/1 | **615** | **188** | 8 / 4 / 8 / 20 / **60** % | 0.1 |

- Full steps fall instead of rising (prediction: ≥ 35 %); fingers-hard and pen iterations rise 2.5–3×. H4 is falsified on its own criterion.
- F3 item 1 (why steps are damped): median ρ is 0.84 at α = 0.03 and 0.46 at α = 1 (all steps: 10/50/90 % quantiles 0.14/0.65/1.5). The model is accurate for the small steps, so curvature is not what limits them. The gain-ratio rule drives reg down (median 0.1–0.5), the Newton step then grows, and the line search cuts it to the smallest α. The table raises reg after a small accepted step and so keeps the step inside the region where the full step is accepted; it acts as a trust-region radius. Per F3: keep the table and describe it in the write-up as a Levenberg–Marquardt rule driven by the accepted step size (X6 stays an argument).
- The T's fewer iterations with the ratio rule (73–94 against 122–199) come with lost seeds in one run; not pursued.

### 11.4 Related results of the same day (`HANDOFF_fidelity_cycles.md` §13)

V15 = V14 + `--lpec-solver fast --lpec-par 3 --lpec-par-lazy 1` (structured QP for the LPEC step, lazy concurrent radii): the stuck-plan set of E6 rescued 7/7 in 410 s against V14's 6/7 in 814 s. This bears on H6: the time of LPEC instances is already roughly halved without a cap on cycles; E6's replay should start from V15 rows. Fine-level round starts (`--lpec-start fine`) were negative there and need not be tried in E6.

### 11.5 Next

T1/E1 (the refactor into `excursion` / `continue_` / `search`, equivalence under `DET=1` against `lfddp_v.py`), then E2 (one stall rule, watchdog everywhere, fine cycles as a rung), E4 (terminal multiplier), E6 (from V15), E7, E5, E8 as planned. Remove `--lm 2` from consideration.
