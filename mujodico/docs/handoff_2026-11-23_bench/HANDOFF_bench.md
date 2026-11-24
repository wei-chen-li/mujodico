# Hard-task benchmark: tool use + assembly (2026-11-23)

User request: implement the tool-use tasks of SimToolReal (Kedia, Lum, Bohg, Liu, RSS 2026,
roboticsproceedings.org/rss22/p151) and the assembly tasks of OmniReset (Yin et al., arXiv
2603.15789, Fig. 1), make sure they are physically solvable, then test the solver. No package
files were edited; everything is in this folder. Build conventions and pitfalls: BUILD_GUIDE.md.

## 1. What the papers do vs. what we model
- **SimToolReal**: 7-DoF iiwa + 22-DoF Sharpa hand (29 DoF), RL policy tracking object-pose
  trajectories from human video; "Task Progress" = fraction of goal poses reached within 2 cm.
  Tools start lying on a table; grasp -> in-hand rotation -> use.
- **OmniReset**: UR7e + Robotiq 2F-85, PPO with diverse resets, IsaacLab/PhysX; tasks from
  FurnitureBench (drawer insertion, leg twisting), peg insertion, cube stacking. No dimensions,
  clearances or numeric success thresholds are given in the paper.
- **Here** (fixed-goal tasks for the repo's solvers, success = terminal metric < 0.01 under the
  TRUE law): floating Allegro hand on a 6-DoF position-servoed wrist (22 actuated DoF) for the tool
  tasks; floating parallel gripper on the same wrist (7 actuated DoF) for the assembly tasks
  (`scripts/embodiments.py`). We score the *functional outcome* (nail depth, screw angle, debris in
  pan, toast flipped, pins pressed, part inserted) rather than tracking a pose trajectory: harder
  for a planner (the tool motion must be discovered) and expressible as a fixed goal.
- **Tool use is enforced**: the hand/gripper cannot touch the target (contype/conaffinity), only
  the tool can.
- **Writing / erasing** need a per-stage reference that the solver does not support, so the
  "ink" is physical: push-pins that stay down. Writing = pressing the pins along an 'L' with the
  pen tip; erasing = pressing a 2x3 grid of raised pins flush with the eraser.

## 2. Tasks (all registered in benchmark_ippas._EXTRA_TASKS by importing scripts/bench_tasks.py)

| Task | file | nu | nx | contacts | lifted cf dims | horizon | do-nothing metric | witness (true law) |
|---|---|---|---|---|---|---|---|---|
| GripperCubeStack | bench_tasks | 7 | 42 | 16 | 64 | 90 x 0.05 s | 5.3 | 8/8 |
| GripperCubeLift (diagnostic) | bench_tasks | 7 | 42 | 16 | 64 | 50 x 0.05 s | 0.64 | 4/4 |
| GripperPegInsert | tasks_gripper | 7 | 29 | 14 | 32 | 90 x 0.05 s | 7.2 | 5/5 |
| GripperDrawer | tasks_gripper | 7 | 29 | 16 | 46 | 120 x 0.05 s | 12.4 | 5/5 |
| GripperLegTwist | tasks_gripper | 7 | 31 | 16 | 34 | 120 x 0.05 s | 11 | 5/5 (+5-7) |
| AllegroHammer | tasks_tools_a | 22 | 59 | 14 | 56 | 150 x 0.04 s | 0.16 | 3/3 |
| AllegroScrewdriver | tasks_tools_a | 22 | 59 | 13 | 52 | 150 x 0.04 s | 2.47 | 3/3 |
| AllegroSpatulaFlip | tasks_tools_a | 22 | 70 | 15 | 60 | 150 x 0.04 s | 1.79 | 3/3 |
| AllegroPenBoard | tasks_tools_b | 22 | 67 | 15 | 45 | 150 x 0.04 s | 0.216 | 3/3 (11/11) |
| AllegroBrushSweep | tasks_tools_b | 22 | 65 | 15 | 60 | 130 x 0.04 s | 1.5 | 3/3 (12/12) |
| AllegroEraserWipe | tasks_tools_b | 22 | 69 | 16 | 46 | 120 x 0.04 s | 0.18 | 3/3 (11/11) |

(For comparison: AllegroPen has 5 contacts / 20 cf dims, nx 45; PushT2dHard 2 dims.)
Witnesses: `python scripts/witnesses.py <Task> a:b --law true` (from repo root mujodico-main-dev,
CPU). Every task also passes a hold test (controls at u_ref: nothing moves > ~1e-3).
Each class docstring lists the model, contacts, cost, tolerance and every simplification.

### Design summaries and known simplifications / exploitable gaps
- **CubeStack**: two 5 cm 100 g cubes; A onto B with yaw +90 deg. Pads-table / pads-B contacts
  dropped (wrist-height floor). Goal fixes one of the cube's symmetric yaws.
- **PegInsert**: 3x3x12 cm peg lying on its side -> grasp near an end, 90 deg wrist roll, insert
  4.5 cm into a hole of two 4-sphere rings (3 mm radial clearance, frictionless walls). Peg can spin
  +-23 deg in the hole; goal picks 1 of 8 symmetric orientations.
- **Drawer**: 7x6x4 cm drawer on its side -> upright, set in front of a 6-sphere cabinet slot
  (4 mm clearance), pushed in. Pads do not collide with the cabinet (a solver may carry it in through
  the walls). Tipping it over did not work in the true law (regularized friction slides, not pivots).
- **LegTwist**: leg with square grip section, round stud and a key pin; fixed round socket; viscous
  nut (hinge) driven by the key between two dog pins; goal nut +90 deg + leg seated. No thread
  advance. Key passes through ring spheres (gaps act as keyway).
- **Hammer**: 60 g head + handle raised by a knob; nail = slide with frictionloss 4 N + damping;
  goal 2 cm depth (+-5 mm). Dropping the hammer head-first from >~6 cm can also drive the nail.
  No in-hand reorientation needed (handle raised).
- **Screwdriver**: handle with bolsters, blade; screw = hinge with frictionloss 0.02 N m, one-sided
  slot of two posts; goal pi/2 (+-0.1 rad). Most fragile: slow wrist rotations + drift correction
  needed; a dropped tool can spin fast enough for the true law to NaN.
- **SpatulaFlip**: toast as 4 spheres (only a crust sphere touchable by the blade); lift the crust
  edge and push the toast over (roll-over goal 8.9 cm along +x, upside down).
- **PenBoard**: thick marker (r 16 mm, inflated inertia), 5 pins in an 'L' at 2 cm, 12 mm proud,
  gravcomp + damped (no frictionloss, which misbehaves on light joints).
- **BrushSweep**: brush with bristles sideways (roll needed), 2 planar crumbs swept to a pan back
  wall; tolerance ~1.4 cm each.
- **EraserWipe**: felt block + handle bar lying on its side, roll ~90 deg, press 2x3 pins flush
  (>= 2 placements). Middle finger excluded from the eraser (budget).
- Common: fingertips have table spheres; finger links / palm / tools are not checked against
  fixtures; costs see only the target, so there is no cost gradient until the tool reaches it.

## 3. True-law pitfalls found while building
1. MuJoCo C is not a proxy at dt 0.04-0.05 (contact time constant floored to 0.1 s: pads sink
   through 100 g cubes); the true law (Hunt-Crossley SAP) is stiff. Certify with `sim_true` only.
2. **Contact count drives GPU memory**: the first cube model (56 contacts, 224 cf dims) made PLF-V
   allocate 12 GB. Budget <= 16 contacts; runs capped with XLA_PYTHON_CLIENT_MEM_FRACTION=0.25.
3. Joint `frictionloss` on light joints is unreliable: saturated friction on untouched pins, which
   then move and jump past their range; a frictional nut blew up after a jammed twist. Use damping
   (pins, nut); hammer/screwdriver keep small frictionloss values that passed hold tests.
4. Frictional contacts at near-zero distance with several pins at once gave forces ~10x outside
   the cone -> NaN; condim 1 for tool-pin and hole/slot walls (also 1 cf dim instead of 4).
5. Box edges/corners sliding past spheres while confined produced phantom impulses.
6. `sim_true` / solver `update` keep the first seed's compiled model: seeds may change only x_0
   and the goal, never geometry.
7. Explicit position gains need omega*dt < ~1.5 (the first jaw servo, kp 2000 on 70 g, exploded).

## 4. Solver tests (PLF-V V15 = vplf.sh BASE + `--mu0 1e-3 --handoff-from incumbent` + V15flags)
Runner: `scripts/run_plfv.sh <Task> <a:b> <label> <out.jsonl>` (wraps the unmodified
handoff_2026-11-22/scripts/lfddp_v.py via scripts/plfv.py; plans in results/plans, logs in
results/logs). `scripts/queue.sh <jobfile>` runs jobs one at a time and waits while the other
session's chain_lad timing window is open.

| Task | seeds | solved | true goal | time | notes |
|---|---|---|---|---|---|
| GripperCubeStack | 0-1 | 0/2 | 0.58, 0.54 | 651, 344 s | drags A on the table next to B; never lifts it over |
| GripperCubeLift | 0-2 | 0/3 | 0.26, 0.34, 0.24 | 295-348 s | never aligns over the cube (4-6 cm off), closes on air, flicks the cube up at the last step |
| GripperCubeLiftGrasped | 0-2 | 0/3 | 0.027, 0.25, 0.08 | 170-477 s | starts holding the cube: opens the grip mid-lift, drops/re-grabs, overshoots 4 cm, final-step pop |
| GripperCubeLiftPregrasp | 0-2 | 1/3 | 0.067, **0.0094**, 0.036 | 84-402 s | squeezes and lifts within 5 steps, then loses the grip; the "solved" seed passes the goal height at +0.6 m/s |

## 5. Why the solver cannot lift a cube (2026-11-23 evening)
Diagnostics on GripperCubeLift (CPU scripts in the session scratchpad: ladder_witness.py, grad_probe.py).
1. **The planning model is not the cause.** The certified witness rolled out in the solver's smoothed model
   (LMS_KW, eps = 2e-5 * sig) reproduces the lift at sig = 1e-3, 1e-2, 0.1, 1 (metric 1e-4 - 3e-4); only sig = 10
   fails (jaws sink in, cube not lifted; the cycles do visit sig 10). Grasped-start plans score the same in the
   smoothed model and the true law (0.027 vs 0.027, 0.252 vs 0.253).
2. **A grasp is a cross term: no first-order signal until it is complete.** |dJ/du| at the pre-grasp (jaws open
   around the cube): grip 1e-12 / 2e-5 / 0.19 and wrist-z 1e-7 / 3e-7 / 0.04 at sig 1e-3 / 1 / 10. Closing alone
   or lifting alone does not move the cube; along the working witness the wrist-z gradient is 0.27-0.81. A bump
   that knocks the cube does lower the cost to first order -> flicks. The only level with strong gradients
   (sig 10) is the level that cannot hold the cube.
3. **The cost does not reward keeping a grasp.** While held, squeezing harder or a bit less changes nothing until
   the cube slips (a cliff at the friction-cone boundary), so the grip command wanders over its whole range.
4. **Terminal-only goal is exploited by tosses** (cube passing through the goal pose at t = T).
Test of 3-4: velocity weight 1 on the cube at the goal (success metric too): goal = at the pose and at rest.
- `...RunStill` (weight also in the stage cost Q): GraspedRunStill 0/3 (0.81, 0.64, 0.054), PregraspRunStill 0/3
  (0.64, 0.023, 0.72). Several seeds end at the do-nothing value 0.64: the stage velocity cost of an 8 cm lift
  (~1.2) exceeds the goal's value (0.64), so not lifting is cheapest. Design error, not evidence against the idea.
- `...Still` (terminal only: Qf and metric_Q), first run: Grasped 0/3 (0.64, 0.74, 1.87), Pregrasp 0/3 (0.65, 0.65,
  0.11), Lift 0/3 (0.63, 0.59, 0.62). Grasped seed 2 ending *worse* than doing nothing exposed the next issue.

### 5.1 All V15 runs above started from a pathological first guess (fixed 2026-11-23 21:00)
lfddp_v.py starts from all-zero controls (`U = zeros`; the warm IP also starts with zero controls). With absolute
position-servo targets, u = 0 commands the wrist to the world origin at the lowest allowed height and the grip
closed: in every task the first trajectory slams the hand/gripper to the floor at the origin, and in the
grasped/pre-grasp diagnostics it drags the cube away (metric 1.6-2.4 vs 0.64 holding still). The repo's earlier
tasks use velocity actuators (u = 0 = stand still) or a fixed hand, so they never had this.
Fix (bench_tasks.register -> _apply_offset, no solver change): every position servo is driven by
force = kp (u + q0 - q) - kv qdot (affine bias biasprm[0] = kp q0, ctrlrange shifted), q0 = the class's seed-0
start pose, so **u = 0 holds the start pose** (u_ref = 0, task.u_offset = q0). The model stays seed-independent;
for pen/brush/eraser (start pose follows the tool) u = 0 holds a home pose <= ~1.5 cm / 0.15 rad from the start.
The pre-grasp/grasped diagnostics now use a fixed cube pose (seeds are repeats). Witnesses keep absolute targets
(simtools.keyframes converts); all 14 re-certified 3/3 with identical metrics. **Every V15 row above is superseded
by the V15h rows** (results/diag_h.jsonl, results/bench_h.jsonl).
Also: the Allegro tasks need > 4 GB GPU memory (2.5 GB single allocations); the first V15 Allegro runs OOM'd under
the 25% cap. V15h Allegro runs use MEMFRAC=0.5, one at a time.

### 5.2 At-rest goal for every task; screwdriver made viscous (2026-11-23 22:00)
Under the pose-only goal every V15h "success" or near-miss on the lift diagnostics was a throw (cube passing
the goal height at +0.56 to +0.95 m/s at t = T). All registered tasks now default to an at-rest goal
(bench_tasks._apply_at_rest: velocity weight 1 per m/s, 0.1 per rad/s on every scored joint, in Qf and the
success metric only). All witnesses re-certified 3/3 except the screwdriver, whose light screw (dry friction
0.02 N m) rattled in the slot after the twist (+-1.7 rad/s at t = T); it is now viscous only (damping 0.02,
no frictionloss) and certifies 3/3 at rest (even 0.01 N m of dry friction brings the rattle back).
Runs under the at-rest goal are labelled V15r (results/diag_r.jsonl, bench_r.jsonl).

### 5.3 Two separate failure modes (CPU probes, 2026-11-23 22:00-22:20)
Probes: scratchpad line_probe.py, lbfgs_probe.py (single-shooting L-BFGS-B over the controls with box
bounds on the smoothed planning objective J = sum stage Q,R + terminal Qf), barrier_probe.py.
- **(B) Once grasped, the problem is easy, but not for PLF-V.** GraspedStill: J falls monotonically along
  the straight line from PLF-V's final plan to the witness, identically in the true law and the smoothed models
  (J 31.3 -> 7.8, metric 0.41 -> 0.0004), so the plan is no local minimum along that line. Naive L-BFGS-B from
  hold-still solves the grasped lift at every smoothing level (sig 1, 0.1, 0.01, 1e-3 and the continuation:
  true metric ~0, J 0.94, 100-230 its, ~1 min on 2 CPU cores). From PLF-V's incumbent L-BFGS stalls at 0.28,
  so PLF-V's first phase (lifted IP exploration, iterations 1-18) steers into a poor basin, and LFDDP then
  reports convergence (|d1| < 1e-4 J) within 1-3 iterations after every model change; its working plan scores
  2.31 in its own finest model vs the incumbent's 0.414. Coarse smoothing is not the cause (L-BFGS at sig 1 is
  fine). The discovery barrier is not the cause either: along plan -> witness the cube-table barrier grows
  (+178 per unit mu) but the pad term falls more (total 3314 -> 3152), i.e. <= 0.2 at PLF-V's mu vs dJ = -23.
- **(A) Before the grasp, no gradient method gets signal.** From hold-still, L-BFGS on Pregrasp and the full
  lift: the projected gradient is exactly 0 at sig 1e-3 (0 iterations, metric 0.64); with continuation from
  sig 1 it ends at 0.60 / 0.64. Cube stack with continuation: 0.59 -> 1.15 (the same drag basin as PLF-V).
- **(A) is mostly the 1 cm far-field cutoff near the object, and an attraction problem far from it.** The planning
  model (reloc_lms.LMS_KW) has far_field_margin = 0.01: contacts more than 1 cm apart give exactly zero force at
  every smoothing level. In the pre-grasp the pads start 1.5 cm from the cube faces. L-BFGS from hold-still with
  continuation sig 1 -> 1e-3: margin 1 cm 0.60, **3 cm 0.0007, 5 cm 0.0006 (solved, at rest)**; at sig 1e-3 alone
  the gradient stays 0 at any margin. From the full-lift start (pads ~8-10 cm away) wider margins (5, 10, 15 cm)
  and coarser starts (sig 10, 100, 1000) do not help: the smoothed force does not reach that far at sane levels,
  and the very coarse models are physically wrong (J 1e5-1e6 at sig 100-1000, cube flung at sig 10). Cube stack
  with a 10 cm margin: 3.17. Approach from far needs an attraction term (PLF-V's discovery barrier provides one;
  L-BFGS has none).
- Summary for the lift, by stage: approach (far) -> needs attraction; close (<= 1.5 cm) -> needs a far-field
  tail > 1 cm; lift once grasped -> easy for single shooting, PLF-V's lifted exploration fails.
- Test queued: PLF-V V15 with far_field_margin 0.05 (scripts/plfv.py env FFM, process-local; the true-law judge
  is unaffected), label V15rF5, results/diag_rf.jsonl.

### 5.4 Why PLF-V's DDP phase cannot do the grasped lift (2026-11-23 23:00-23:50)
Ablations on GripperCubeLiftGrasped (at rest, fixed cube; 3 repeats; scripts/run_plfv_x.sh + queue_x.sh replace
the BASE flags; lfddp_v takes the FIRST occurrence of a flag):
| variant | result |
|---|---|
| full PLF-V V15r | 0.053, 0.116, 0.136 |
| X1 LFDDP alone (--warm none), smoothing ladder | 0.58 x3 |
| X2 LFDDP alone at sig 1e-3 | 0.63 x3 |
| X3 = X1 with the discovery barrier off (mu0 1e-9) | 0.55 x3 |
| X4 = X2 with the barrier off | 0.59, 0.60, 0.60 |
| verbose X1 + open-loop forward pass (bench copy lfddp_v_bench.py, --kscale -1) | 0.605 |
| verbose X1 + lagged dynamics curvature (--curv 1) | 0.598 |
| naive L-BFGS-B on the same smoothed objective (with or without the barrier, any sig) | ~0 (solved) |
- The lifted IP exploration is not the culprit (LFDDP alone is worse), nor the barrier, the feedback in the
  forward pass, or Gauss-Newton vs lagged curvature.
- Mechanism seen in `--verbose 1` traces: a burst of progress after each model change (J 162 -> 48 in 7 its),
  then every step is rejected; the Levenberg-Marquardt regularization climbs to its 1e6-1e8 cap and LFDDP
  either crawls (dJ ~0.03/it) or gives up after 8 failed tries -- which it logs as "converged".
- At reg 1e8 the predicted decrease is still ~0.75; for a regularized step d1 ~ |Qu|^2 / reg, i.e. LFDDP's
  control gradient is ~1e4 in norm, while the autodiff gradient of the same objective at the start is 2 (sig
  1e-3) - 72 (sig 1) with negative curvature (-2e3, -2.4e4) along -grad. LFDDP's local model predicts decreases
  that the rollout does not deliver even for tiny steps -> its linearization (A = f_x + f_cf Phi_x, contact
  forces eliminated) disagrees with the objective's true sensitivity on this grasp. **Superseded by the next
  bullet** (the inflated-gradient reading was wrong).
- **Predicted vs actual decrease per step size** (bench copy, `--verbose 2`, open loop, barrier off): at the
  accepted step the model is accurate (actual/predicted 0.83-1.3), but the next larger step falls off a cliff:
  e.g. it 12, a = 0.25 predicted -0.0095, actual -0.0099; a = 0.5 predicted -0.019, actual **+8.2e5**. A jump of
  that size is LFDDP's velocity penalty 1e3 (|v| - 5)^2: a slightly larger control change makes the cube squirt
  out of the two-sided squeeze at high speed (a stiff knife-edge equilibrium). The Levenberg-Marquardt rule
  (--lm 1) raises the regularization whenever the accepted step is < 1, so reg ratchets to its cap and DDP
  crawls. Without the ratchet (--lm 0) it does a little better (0.48) but still stalls at the cliff, and the next
  terminal-weight boost sends J to 1.9e6. L-BFGS has the same cliffs in its objective but its steps adapt both
  ways and it uses 100-300 cheap iterations.
- So for the grasped lift: the obstacle is the squirt cliff of a stiff point-contact pinch, which DDP's
  step/regularization logic handles badly. Candidates: (task) more compliant pads / more pad contact points to
  soften the cliff; (solver) a single-shooting quasi-Newton polish phase, or a trust region in control space
  whose radius adapts independently of the LM regularization.

### 5.5 Pad variants against the squirt cliff (2026-11-24 00:00, running)
Full PLF-V without the barrier (X5): 0.085, 0.12, 0.11 (with barrier 0.053-0.136) -> barrier ruled out.
Gripper options (embodiments.gripper_scene pad_x, pad_solref; bench_tasks.PAD_VARIANTS, suffixes on every lift
diagnostic and GripperCubeStack):
| tag | pads | contact stiffness (efc K) | contacts / cf dims | squeeze penetration |
|---|---|---|---|---|
| (base) | 2 spheres per pad | 111 | 16 / 64 | 0.10 mm |
| S2 | soft, solref 0.2 s | 28 (4x softer) | 16 / 64 | 0.23 mm |
| S4 | soft, solref 0.4 s | 6.9 (16x softer) | 16 / 64 | 0.91 mm |
| P | 2 x 2 grid per pad (+-6 mm across) | 111 | 20 / 80 | 0.10 mm |
| PS2 | grid + soft 0.2 s | 28 | 20 / 80 | 0.12 mm |
MuJoCo averages the two geoms' solref unless one has priority: the pad spheres get priority="1" (without it,
0.2 s pads mixed to ~0.11 s, barely softer). All variants' witnesses certify 2/2 (lift diagnostics + stack).
PLF-V runs: results/pads.jsonl (label V15r), with same-conditions baseline reruns (label V15rB) for the timing
comparison.

### 5.6 ROOT CAUSE: PLF-V's planning model omits non-contact constraints (2026-11-24 01:00-02:00)
**Everything in 5.3-5.5 about "DDP vs L-BFGS" was confounded by this.** The lifted transcription
(trajopt/_src/transcriptions.py lifted_shooting_transcription) applies only the lifted CONTACT forces in its dynamics;
joint/tendon limits, equality constraints and friction loss exist only in `true_step` (its docstring: "the two agree
whenever no non-contact constraint is active"). IppasLiftedMultipleShooting.rollout uses true_step
(full_law_rollout=True), so my earlier "smoothed model reproduces the grasp" check and the L-BFGS probes used the
full law, while PLF-V's lifted IP and LFDDP plan with the lifted law.
- Gripper: the jaws were coupled by a joint equality (jr = jl, one actuator): in the lifted law the right jaw is an
  unactuated free slider (witness: jr flies to 1.16 m, cube dropped, metric 4.33 vs true 0.0004). No pinch exists in
  PLF-V's model. That is the "squirt cliff" of 5.4.
- Allegro: every witness relies on the thumb-base joint stop (thj0 at 1.396 under the squeeze); in the lifted law the
  thumb swings past it and the grasp falls apart. Witness metrics, true vs lifted: hammer 0.0000 vs 27, screwdriver
  0.0000 vs 8.1/NaN, spatula 0.002 vs 1.8, pen 0.0000 vs 0.216 (= do-nothing), brush 0.002 vs 1.1, eraser 0.0000 vs
  0.18. Pins / nail also rely on joint stops; the drawer witness on the wrist-height floor (joint limit).
- Controlled DDP test (scripts/ilqr.py, minimal box-iLQR, same cost): on the lifted law iLQR diverges on the grasped
  lift; on the full law (--law full) it solves it in 5 iterations (sig 1) / 10 (sig 1e-3). DDP is fine.
**Fixes**
1. Gripper embodiment (embodiments.gripper_scene coupled=False, now the default): two independently servoed jaws
   (a_grip, a_grip_r; nu 8). No equality constraint, so the unpatched lifted law represents it. simtools.keyframes
   mirrors a_grip to a_grip_r for the witnesses. Lifted == true on the cube tasks, peg, leg (not the drawer: floor).
   **Unpatched PLF-V V15 (label V15j): GraspedLift 3/3 (49 its, 4-5 s/seed after compile), PregraspLift 3/3 (84-86
   its, 8-13 s/seed)** -- previously 0/3 and 1/3 at 200-650 s/seed. CubeStack still 0/2 (0.41, 0.42).
2. Formulation (revertible, process-local): scripts/fulllaw_patch.py v2 lifts EVERY constraint force -- the lifted
   vector is [contacts | far-field candidates | non-contact constraint forces (limits, equalities, friction loss)]; the
   consistency equality covers all of them (one solve, runtime smoothing) and the dynamics apply all of them, so at
   consistency the lifted law == true_step. New entries are unbounded (no barrier); contact-indexed code (barrier map,
   cf_index) is unaffected. Enable with FULLLAW=1 in plfv.py (also sets IPPAS_EXPORT='', since the export cache keys on
   the package source). Witness check, patched lifted vs true: all 10 tasks agree (e.g. hammer 0.0000/0.0000, pen
   0.0000/0.0000, drawer 0.0592/0.0591); rollout cost unchanged. v1 (non-contact forces from a second solve) did not
   reproduce pins-on-stops and is kept as fulllaw_patch_v1_noncontact_from_full_solve.py. Runs: label V15L,
   results/full.jsonl.

### 5.7 What still fails after the fix: discovery from a far start (2026-11-24 03:00-05:00)
With the model fixed, every gripper start that is within ~5 cm of a pre-grasp solves (GraspedLift 3/3, PregraspLift
3/3, offset pre-grasps X2 / Yaw3 / X2Yaw3 / Z5 8/8, 3-24 s/seed after compile). What fails is the full task from the
real start (wrist 15 cm above, cube up to 14 cm away in xy): GripperCubeLift 0/3 under every variant (V15j, V15L,
far-field margin 3 / 5 / 10 cm), GripperCubeStack 0/2 everywhere (best 0.36-0.38 with a 10 cm margin).
- Horizon slack (GripperCubeLiftLong, 70 steps vs 50; witness needs 44): **seed 0 solved (0.0067, 638 its, 373 s) --
  the first full lift from the real start**; seeds 1, 2 not (0.578 = cube untouched). Replays (true law) of the failed
  plans: the wrist hovers 5-14 cm above the cube or 5 cm beside it in y -- the jaws never straddle it. The coarse
  level's own plan never lifted (plan goal 0.58-0.64 = start), so this is a plan-level stationary point, not a
  smoothed-vs-true gap. GripperCubeStackLong 0/2 (0.80, 0.67; with 10 cm margin 0.43, 0.38).
- Gradient probe (scratch: gradient of the smoothed planning objective at the hold-still plan, GripperCubeLiftLong
  seed 2): sig 1: |g| 1.1e-2 at J 69; sig 0.1: |g| 9e-5 -- flat, identical for far-field margin 1 cm and 10 cm (the
  pads are 9-15 cm away). sig 10: |g| 42, but J rises 69 -> 84 at the SAME plan and the gradient moves the wrist AWAY
  from the cube in x and z; sig 100: J ~ 1e7 (non-physical). For comparison the aligned pre-grasp 5 cm above has
  |g| 0.13-0.23 at sig 1, dominated by the grip.
- Reading: the smoothed law's forces are push-only normals (+ friction inside the cone). A distant pad can only shove
  the cube down/away, which the lift goal penalizes, so the far-field signal is zero (physical smoothing) or repulsive
  (coarse smoothing). Pushing tasks (PushT etc.) work because their far-field push already points at the goal; a lift
  or pull has no far-field representative. This is a property of smoothing-based discovery, not of DDP: L-BFGS (SSQN)
  and plain iLQR stall at the same start (0.638 = untouched).
- Candidate generic fix being tested: planning-only VIRTUAL FORCES (scripts/vfilqr.py, minimal iLQR): every free body
  that has a robot contact pair gets a 3D force control, penalized w (dmin + delta)|v|^2 with dmin the soft-min of the
  robot-object pair distances MJX already reports (true distances for every pair in the contact list); w is raised
  (1, 10, 100, 1000) and then v := 0 for the usual sigma ladder; read-outs use u only. The penalty's cross term pulls
  the robot toward whatever object it wants to move, in any direction. Nothing task-specific.
  **Result: negative (GripperCubeLiftLong seed 2, 3 configs, results/vf.jsonl, 3-6 min each on CPU).** VF1 (soft-min
  distance, w 1/10/100/1000, sig 1): w=1 brings a pad within 6 mm (planning J 69 -> 40) but by sitting on the cube top
  while the virtual force lifts; at w >= 100 the iLQR stalls, and with v := 0 it returns to hold (true 0.62). VF2/VF3
  (mean over robot geoms, w 1..1000 in x3 steps, sig 0.1 / 1): same; the w=3 plan JUGGLES the cube to the goal with
  alternating virtual forces while the wrist swings +-1.5 rad and the pads stay 3-15 cm away. Every phase after the
  first stalls at iteration 1: the plan sits on a knife edge (random control perturbations of 1e-4 change J smoothly,
  1e-2 by +2700; along the Newton step even a = 1e-4 jumps J by +400 closed-loop / +6000 open-loop although |k| <=
  0.013 -- a contact-geometry discontinuity, pads inside the box). Lesson: a free-floating virtual force is too
  strong a relaxation -- it lets the object move without the robot. The next candidate is an ADHESION relaxation
  (the virtual force acts between a robot geom and the object, equal and opposite, penalized by their distance), so
  the robot must carry what it moves.
  **Adhesion variant (--mode adhesion), also negative.** Controls s >= 0 per robot-object contact pair; the object is
  pulled toward that robot geom along MJX's pair normal with magnitude s exp(-d / ell), applied as qfrc_applied
  inside a smoothed full MJX step (so contacts respond; the first version, a velocity kick after the step, tunnels);
  penalty w s^2. VA5 (ell 5 cm) and VA10 (ell 10 cm), w 0.1 -> 100 in x3 steps: the early phases do pull the pads
  onto the cube (planning J 69 -> 50, dmin ~0) but by suction from the top, and as w rises the iLQR smoothly backs
  off to hold (dmin 6-7 cm, true 0.638) instead of switching to a pinch. Checked that this is not an iLQR bug: the
  forward pass equals a plain rollout of the same controls bit for bit; the stalls come from the planning dynamics
  itself -- in the VF2 plan a 1.3e-6 control change moves the state by 7.5 at step 10 (all earlier steps < 1e-4),
  i.e. the relaxed plans drive pads deep into the box where the smoothed contact geometry is discontinuous.
  **Conclusion:** continuation from a relaxed "the object follows the robot" plan does not lead into the grasp: the
  path of least resistance out of the relaxation is to let go. Pick-and-place from far needs the contact MODE (pinch)
  to be discovered, which none of the smooth relaxations tried here (far-field margin, coarse smoothing, free
  virtual forces, adhesion) provides. Remaining generic options, none tried: (a) a coarse-level search over contact
  modes / initial guesses (excluded for now by the no-sampling rule), (b) a reaching-style shaping term (robot near
  the objects it must move) -- simple but task-shaping, (c) longer horizons + more seeds (LiftLong: 1/8 seeds, so seed 0 was
  luck, not a reliable fix).

## 6. Results log
All PLF-V runs: V15 flags; true-law metric < 0.01 = solved; time per seed includes compile on the first seed.
Rows are in results/*.jsonl (label column); the most informative after the model fix:

| task (variant) | label (model) | solved | best metrics | notes |
|---|---|---|---|---|
| GripperCubeLiftGrasped | V15j (indep. jaws) / V15L (patched) | 3/3, 3/3 | 0.0008-0.0016 | 47-49 its, 3-5 s/seed |
| GripperCubeLiftPregrasp | V15j / V15L | 3/3, 3/3 | 0.0011-0.0026 | 84-88 its, 4-13 s/seed |
| Pregrasp offsets X2, Yaw3, X2Yaw3, Z5 | V15j | 8/8 | 0.0003-0.0029 | 55-319 its, 3-24 s/seed |
| GripperCubeLift (15 cm above, H50) | V15j, V15L, FFM 3/5/10 cm | 0/15 | 0.52-0.58 | cube untouched |
| GripperCubeLiftLong (H70) | V15j | 1/8 | s0 0.0067; others 0.51-0.58 | seed 0: 638 its, 373 s |
| GripperCubeLiftLong (H70) | V15jF10 (10 cm margin) | 0/3 | 0.53-0.58 | |
| GripperCubeStack (H90) | V15j / V15L / FFM 10 | 0/2 each | 0.36-0.55 | |
| GripperCubeStackLong (H120) | V15j / V15jF10 | 0/2, 0/2 | 0.67-0.80 / 0.38-0.43 | |
| GripperPegInsert | V15L | 0/2 | 1.05 | |
| GripperLegTwist | V15L | 0/2 | 3.14 | no rotation |
| GripperDrawer | V15L | 0/2 | 0.34, 0.42 | was 2.1-2.3 before the fixes |
| AllegroEraserWipe | V15L (patched) | 1/2 **INVALID** | s1 0.007, s0 0.030 | 3700-4700 s/seed. Both plans toss the eraser: it tumbles over the pins, then rests upside down with handle+block 25 mm inside the table (only its 3 feet collide with the table). Task bug, see 5.8 |
| AllegroSpatulaFlip | V15L (patched) | 0/1 | s0 0.906 (s1 crashed: cuSolver handle, GPU full; requeued in R4) | 2340 s |
| AllegroBrushSweep | V15L (patched) | 0/2 | s0 0.044, s1 0.194 | 1360-6480 s/seed |
| AllegroHammer, PenBoard, Screwdriver | V15L | -- | -- | 06:00 reruns crashed again: hammer RESOURCE_EXHAUSTED (ran beside brush), pen cuSolver handle failure at start (GPU full), screwdriver NaN passed into the LPEC (Gurobi "Nan or Inf" in lpec_step._grb) after 20 min. Rerun 08:43 serially alone at MEMFRAC 0.6 (queue_R4); crash logs kept as log_*_crash1.txt |
| controls: minimal iLQR (ilqr.py) | lifted / full law | GraspedLift 1/1 in 5 its (full law) | | DDP fine on the right model |
| virtual-force / adhesion iLQR (vfilqr.py) | LiftLong seed 2 | 0/5 configs | 0.62-0.64 | see 5.7 |



### 5.8 Two findings, 2026-11-24 09:00

**Tool/table contact leak (task bug) -- FIXED (scripts/table_contacts.py, BENCH_TABLE=legacy restores).** Every free
object touched the table only through helper spheres placed for its START orientation; a tossed / flipped object sank
into the table. PLF-V's eraser "solve" (V15L seed 1, 0.007) exploited this: toss, tumble over the pins, sink 25 mm.
Fix, applied at model compile time in embodiments.py for every task:
- table box -> plane at z = 0 (same surface);
- the objects' real geometry collides with it (private contype bit 1<<20): plane-box 4 contacts, plane-capsule 2,
  plane-cylinder 3; helper spheres that only supported the start pose lose their table contact. Kept: capsule end-cap
  spheres (pen, brush handle), knobs / handle-end spheres larger than the handle, toast mid-plane spheres, and the
  hidden stand-off feet of the eraser and brush head;
- MJX's plane_convex (plane-box) reports only the vertices within 1 mm of the deepest one and marks the rest invalid
  (dist 1). The laws here need every listed contact with its TRUE distance (sphere/capsule colliders give that); with
  dt 0.05 a slightly tilted peg lost its far-end support for a whole step and rotated through the plane (78 mm drift
  in 90 zero-control steps). table_contacts installs a process-local plane-box collider returning the 4 lowest of the
  8 vertices with true distances (shared mjx source untouched);
- brush head (bristles) friction 0.25 with priority 1: as a rigid box with the table's 0.6 the head corner snagged at
  touch-down and the witness failed (0.65); with 0.25 it solves 3/3 (0.0003-0.0006, better than legacy).
Checks (scripts/check_table.py, check_flip.py): non-table pairs identical in all 11 models; rest drift <= 0.4 mm;
all witnesses (3 seeds each) still < 0.01; flipped objects dropped from 5 cm settle <= 0.5 mm into the table
(legacy: cubes 42, hammer 31, drawer 29, eraser 25, brush 13, spatula 12 mm).
Contacts: cubes 16 (same), peg 14->16, leg 16->19, drawer 16->14, hammer 14->17, screwdriver 13->14, spatula 15->19,
pen 15->16, brush 15->19, eraser 16->22. All earlier results are on the leaky models; V15T reruns (table.jsonl,
queue_TA/TG) started 2026-11-24 09:19.

**Allegro solves are slow because of OpenBLAS thread oversubscription.** The CPU DDP backward pass (lfddp_v backward:
Riccati + _boxqp, jaxlib LAPACK = scipy-openblas) costs 0.7-1.4 s per pass on Allegro tasks (nx 59-70, nu 22, N 120-150)
vs 2-7 ms on the gripper (nu 7). Isolated replica (scratch bwd_bench.py, random data, machine loaded):
default 1713 ms; XLA eigen single-thread 1898 ms; taskset 4 cores 13.8 ms; OPENBLAS_NUM_THREADS=1 22 ms.
The solver process ran ~300 threads at ~1000% CPU: every one of the 16 CPU devices' LAPACK calls spawns a busy-waiting
OpenBLAS pool. nu 7 stays under OpenBLAS's threading threshold, nu 22 does not. bwd was 47-71% of Allegro wall time.
run_plfv.sh now exports OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 (revert: delete that line). Restarted hammer: 124% CPU,
load 35 -> 5.5. End-to-end speedup pending (queue_R4).
