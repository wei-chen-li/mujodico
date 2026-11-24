# Building benchmark tasks (2026-11-23)

Goal: fixed-goal trajectory-optimization versions of the SimToolReal (RSS 2026) tool-use tasks
and the OmniReset (arXiv 2603.15789) assembly tasks, each **certified physically solvable** by a
scripted open-loop witness under the solver's TRUE law.

## Files (scripts/)
- `embodiments.py` — `allegro_scene/allegro_model` (Allegro on a 6-DoF position-servoed wrist,
  thumb tip collides), `gripper_scene/gripper_model` (parallel gripper, 1 grip actuator, jaws
  coupled by equality). Wrist joints `wx wy wz wrx wry wrz`, actuators `a_wx ... a_wrz`; Allegro
  finger actuators `ffa0..3 mfa0..3 rfa0..3 tha0..3` (joints `ffj0..`); gripper `a_grip`
  (target half-opening, 0 = closed, max 0.0425).
- `bench_tasks.py` — shared helpers (`register`, `pose_Q`, `_qpos_adr`, `yaw_quat`,
  `_corner_spheres`, `problem_size`), `GripperCubeStack` (the reference example), and imports
  the group modules `tasks_gripper`, `tasks_tools_a`, `tasks_tools_b`.
- `simtools.py` — `sim_true(task, us)` (TRUE law, CPU), `metric(task, x)`, `keyframes(task, kf)`.
- `handik.py` — fingertip IK (`ik`, `tip_geom`, `FINGER_JOINTS`, `tip_positions`).
- `witnesses.py <Task> a:b --law true` — runs `task.witness()` (or the WITNESS dict) and reports.

## Conventions
- Task = `bench_tasks.BenchTask` subclass, decorated `@register`, with a `witness(self)` method
  returning open-loop controls `(N, nu)` (usually via `keyframes`). Seeds randomize the start.
- Cost: quadratic on the goal state, typically object pose `pose_Q(m, joint, w_pos=100, w_rot=1)`
  or a joint coordinate; `Qf = 40 Q`; success = terminal metric < 0.01 (100*e^2 < 0.01 -> 1 cm).
  `u_ref` = initial actuator targets; `R` small (1e-2 lin, 1e-3 ang, 1e-2 fingers).
- `contact_smoothing=0.002`, `contact_approx` default (Hunt-Crossley).
- **Contact budget: <= 16 contacts** (`problem_size(task)[2] <= 64`; 4 dims per condim-3 contact).
  56 contacts made the solver allocate 12 GB of GPU memory. Spend contacts only where they
  matter; replace never-binding contacts with joint ranges (e.g. wrist-height floor) and say so
  in the docstring. Use spheres (sphere-sphere, sphere-box, sphere-capsule give far-field
  distances); avoid box-box and mesh. Use `bottom_only` corner spheres for objects that stay upright.
- **Tool use must be necessary**: exclude hand/gripper contact with the target (nail, debris,
  toast, screw, ink pins) via contype/conaffinity so only the tool can act on it. Document it.
- Joint dry friction (`frictionloss`) and damping are supported by the true law (used for
  nails, screws, pins).
- Start objects resting 1e-4 into their support (the unsmoothed law is singular at exactly 0).
- MuJoCo C (`sim_c`) is NOT a faithful proxy at dt 0.04-0.05 (contacts floor to a 0.1 s time
  constant -> pads sink through objects); certify with `sim_true` only (first call compiles,
  ~6 s gripper / ~20 s Allegro; then ~30 ms per rollout).
- Explicit position gains: keep `omega*dt < ~1.5` (kp/m_eff); the grip servo uses armature 0.5.

## Proven primitives
- Gripper pick: wrist above object at the object's yaw (jaws close along wrist y), descend so the
  pad spheres are at object mid-height (pad spheres at z = -0.045, -0.025 below the wrist), close
  (`a_grip` -> 0), lift, move, place, open. Holds 100 g cubes stiffly in the true law.
- Allegro palm-down pinch (runnable: scripts/example_allegro_lift.py, run from the repo root mujodico-main-dev) of a handle lying on the table along world y: palm quat in wrist frame
  `quat_mul(axis_quat([1,0,0], pi), [0,1,0,1]/sqrt2)` (fingers point +x and flex toward -z, thumb
  on the +y side). Thumb base `thj0`=1.4. IK the three finger tips to the handle's +x side and the
  thumb tip to its -x side at handle height (+4 mm); squeeze = IK with targets 1.5 cm past the
  surface; pre-grasp = IK 3 cm outward. Handle r=15 mm, 64 g lifts and follows wrist rotation
  (see scratch test in the main session); the handle rolls ~40 deg about its axis while squeezed.
