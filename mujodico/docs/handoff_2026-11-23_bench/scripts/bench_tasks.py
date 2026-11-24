"""Hard-task benchmark (2026-11-23): tool use (SimToolReal, RSS 2026) and assembly (OmniReset,
arXiv 2603.15789), rebuilt as fixed-goal trajectory-optimization tasks for this repo's solvers.

Every task is a `Task` (fixed goal state, quadratic running/terminal cost, success = terminal
metric < 0.01 under the TRUE law) and is registered in benchmark_ippas._EXTRA_TASKS on import.
Each has a scripted witness (witnesses.py) whose open-loop controls reach the goal under the
true law: the certificate that the task is physically solvable within the horizon and the
actuator limits.

Gripper tasks (OmniReset): GripperCubeStack, GripperPegInsert, GripperDrawer, GripperLegTwist
Allegro tasks (SimToolReal): AllegroHammer, AllegroScrewdriver, AllegroBrushSweep,
                             AllegroSpatulaFlip, AllegroPenBoard, AllegroEraserWipe
"""
import numpy as np
import mujoco
from mujodico.examples import benchmark_ippas as B
from mujodico.examples.tasks.task import Task
import embodiments as E
from embodiments import fmt

TASKS = {}


# ---------------------------------------------------------------------------------------------
# Controls relative to the start pose.  The solvers start from all-zero controls; with absolute
# position-servo targets that is a lurch of the wrist to the world origin at the floor (found
# 2026-11-23: it dragged the cube away in the grasped-start diagnostics and made every first
# trajectory a slam to the table).  Every registered task therefore drives its position servos with
#   force = kp (u + q0 - q) - kv qdot      (affine bias biasprm[0] = kp q0, ctrlrange shifted by -q0)
# so u = 0 holds q0.  q0 = the class's seed-0 initial actuator targets (the model must not depend on
# the seed: the solver compiles it once), so for tasks whose start pose varies with the seed u = 0
# holds a fixed home pose near the start.  task.u_offset = q0; witnesses are written in absolute
# targets and converted by simtools.keyframes.
# ---------------------------------------------------------------------------------------------
_Q0 = {}
_COMPUTING = set()


def _apply_offset(task):
  cls = type(task)
  if cls in _COMPUTING:
    return
  if cls not in _Q0:
    _COMPUTING.add(cls)
    try:
      _Q0[cls] = np.array(cls(0).u_ref, float)
    finally:
      _COMPUTING.discard(cls)
  q0 = _Q0[cls]
  m = task.model
  for i in range(m.nu):
    kp = m.actuator_gainprm[i, 0]
    assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE and np.isclose(m.actuator_biasprm[i, 1], -kp) \
        and m.actuator_biasprm[i, 0] == 0.0, f'actuator {i} is not a plain position servo'
    m.actuator_biasprm[i, 0] = kp * q0[i]
    m.actuator_ctrlrange[i] -= q0[i]
  object.__setattr__(task, 'u_offset', q0)
  object.__setattr__(task, 'u_ref', np.asarray(task.u_ref, float) - q0)


# ---------------------------------------------------------------------------------------------
# Goal = at the pose AND at rest (default for every registered task, 2026-11-23).  With a pose-only
# terminal goal the solver's best plans were throws (the cube passing through the goal height at
# 0.6-1 m/s at t = T).  Every joint whose position is scored gets a velocity weight at the goal
# (Qf and the success metric only; the stage cost Q stays pose-only -- a stage velocity cost made the
# solver not lift at all): 1 per (m/s) on translational dofs, 0.1 per (rad/s) on rotational ones.
# ---------------------------------------------------------------------------------------------
V_LIN, V_ANG = 1.0, 0.1


def _apply_at_rest(task):
  m = task.model
  base = np.array(getattr(task, 'metric_Q', task.Q), float, copy=True)
  Qf = np.array(task.Qf, float, copy=True)
  scored = np.diag(task.Q)[:m.nq] > 0
  added = []
  for j in range(m.njnt):
    t = int(m.jnt_type[j]); qa = int(m.jnt_qposadr[j]); da = m.nq + int(m.jnt_dofadr[j])
    nq_j = {0: 7, 1: 4, 2: 1, 3: 1}[t]
    if not scored[qa:qa + nq_j].any():
      continue
    ws = {0: [V_LIN] * 3 + [V_ANG] * 3, 1: [V_ANG] * 3, 2: [V_LIN], 3: [V_ANG]}[t]
    for i, w in enumerate(ws):
      if base[da + i, da + i] == 0:
        base[da + i, da + i] = w; Qf[da + i, da + i] += 40 * w; added.append(m.joint(j).name)
  object.__setattr__(task, 'metric_Q', base)
  object.__setattr__(task, 'Qf', Qf)
  object.__setattr__(task, 'at_rest_joints', sorted(set(added)))


def register(cls):
  TASKS[cls.__name__] = cls
  B._EXTRA_TASKS[cls.__name__] = cls
  orig = cls.__dict__.get('__init__')
  if orig is not None:
    def __init__(self, seed=0, _orig=orig):
      depth = getattr(self, '_init_depth', 0)
      object.__setattr__(self, '_init_depth', depth + 1)
      _orig(self, seed)
      object.__setattr__(self, '_init_depth', depth)
      if depth == 0:
        _apply_offset(self)
        if getattr(type(self), 'AT_REST', True):
          _apply_at_rest(self)
    __init__.__doc__ = orig.__doc__
    cls.__init__ = __init__
  return cls


def _diag(nx, entries):
  Q = np.zeros((nx, nx))
  for i, v in entries.items():
    Q[i, i] = v
  return Q


def _qpos_adr(m, joint):
  return int(m.jnt_qposadr[m.joint(joint).id])


def _qvel_adr(m, joint):
  return m.nq + int(m.jnt_dofadr[m.joint(joint).id])


def yaw_quat(a):
  return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def pose_Q(m, joint, w_pos=100.0, w_rot=1.0, w_vel=0.0):
  """Diagonal weight on a free joint's position (w_pos), quaternion (w_rot) and velocity (w_vel)."""
  nx = m.nq + m.nv
  a = _qpos_adr(m, joint); v = _qvel_adr(m, joint)
  ent = {a + i: w_pos for i in range(3)}
  ent.update({a + 3 + i: w_rot for i in range(4)})
  if w_vel:
    ent.update({v + i: w_vel for i in range(6)})
  return _diag(nx, ent)


class BenchTask(Task):
  """Helpers shared by the benchmark tasks."""

  def wrist_x(self):
    m = self.model
    return np.array([self.x_0[_qpos_adr(m, j)] for j in E.WRIST_JOINTS])



# =============================================================================================
# OmniReset-style gripper tasks
# =============================================================================================
TABLE = """    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" rgba="0.55 0.45 0.35 1"
          contype="2" conaffinity="{ca}" friction="0.6"/>"""


def _corner_spheres(name, h, r, ct, ca=0, bottom_only=False, rgba='0.2 0.2 0.8 0'):
  out = []
  zs = (-1,) if bottom_only else (-1, 1)
  for sx in (-1, 1):
    for sy in (-1, 1):
      for sz in zs:
        p = (sx * (h[0] - r), sy * (h[1] - r), sz * (h[2] - r))
        out.append(f'      <geom name="{name}_c{sx:+d}{sy:+d}{sz:+d}" type="sphere" size="{r}" pos="{fmt(p)}" mass="0" '
                   f'contype="{ct}" conaffinity="{ca}" rgba="{rgba}" group="3"/>')
  return '\n'.join(out)


CUBE_H = 0.025   # half size (5 cm cubes)
PAD_LOW = 0.045 + 0.006   # lowest pad point below the wrist origin (pads vertical)


@register
class GripperCubeStack(BenchTask):
  """OmniReset 'Cube Stacking': stack cube A on cube B with a desired orientation.

  Parallel gripper on a 6-DoF wrist; two 5 cm, 100 g cubes on a table.  Goal: A centred on B's
  top face with A's yaw = B's yaw + 90 deg, B undisturbed.  Seeds randomize both cubes' (x, y, yaw).
  Contacts (16, all sphere-box): 4 pad spheres vs A; A's 4 bottom corner spheres vs table and
  vs B; B's 4 bottom corner spheres vs table.  Not modelled (documented simplifications): pads
  vs table (the wrist-height floor wz >= PAD_LOW keeps vertical pads above it), pads vs B,
  A's top corners.  Cost: A pose (pos 100, quat 1) + B pose (pos 100, quat 0.1); success < 0.01
  (~1 cm, ~11 deg)."""
  HORIZON = 90
  DT = 0.05
  GRIP_LO = 0.0     # lowest grip command (0 = jaws closed); < 0 allows commanding past closed
  PAD_X = (0.0,)    # pad sphere columns across the pad width (2 rows x len(PAD_X) spheres per pad)
  PAD_SOLREF = None # pad contact (timeconst, dampratio); None = default (floored to 2 dt = 0.1 s)

  def __init__(self, seed=0):
    h = CUBE_H
    rng = np.random.default_rng(5000 + seed)
    pa = np.array([0.10 + rng.uniform(-0.03, 0.03), 0.08 + rng.uniform(-0.03, 0.03), h - 1e-4])
    pb = np.array([-0.08 + rng.uniform(-0.03, 0.03), -0.06 + rng.uniform(-0.03, 0.03), h - 1e-4])
    ya, yb = rng.uniform(-0.5, 0.5, 2)
    # bits: pads ct 1; table ct 2 (ca 0); A box ca 1; A corners ct 4 -> table (ca 4), B box (ca 4); B corners ct 8 -> table (ca 8)
    env = f"""    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" rgba="0.55 0.45 0.35 1"
          contype="0" conaffinity="{4 | 8}" friction="0.6"/>
    <body name="cubeA" pos="{fmt(pa)}">
      <freejoint name="cubeA"/>
      <geom name="cubeA" type="box" size="{h} {h} {h}" mass="0.1" rgba="0.85 0.3 0.25 1" contype="0" conaffinity="1" friction="0.8"/>
{_corner_spheres('cubeA', (h, h, h), 0.004, ct=4, bottom_only=True)}
      <geom name="cubeA_mark" type="box" size="{h * 0.6} 0.003 0.0005" pos="0 0 {h}" contype="0" conaffinity="0" mass="0" rgba="1 1 1 1"/>
    </body>
    <body name="cubeB" pos="{fmt(pb)}">
      <freejoint name="cubeB"/>
      <geom name="cubeB" type="box" size="{h} {h} {h}" mass="0.1" rgba="0.25 0.5 0.85 1" contype="0" conaffinity="4" friction="0.8"/>
{_corner_spheres('cubeB', (h, h, h), 0.004, ct=8, bottom_only=True)}
      <geom name="cubeB_mark" type="box" size="{h * 0.6} 0.003 0.0005" pos="0 0 {h}" contype="0" conaffinity="0" mass="0" rgba="1 1 1 1"/>
    </body>"""
    ranges = ((-0.3, 0.3), (-0.3, 0.3), (PAD_LOW, 0.35), (-1.6, 1.6), (-1.6, 1.6), (-3.2, 3.2))
    m = E.gripper_model(env, wrist_pos=np.zeros(3), ranges=ranges, timestep=self.DT, grip_lo=self.GRIP_LO,
                        pad_x=self.PAD_X, pad_solref=self.PAD_SOLREF)
    nq, nv = m.nq, m.nv
    x0 = np.zeros(nq + nv)
    a, b = _qpos_adr(m, 'cubeA'), _qpos_adr(m, 'cubeB')
    x0[a:a + 3] = pa; x0[a + 3:a + 7] = yaw_quat(ya)
    x0[b:b + 3] = pb; x0[b + 3:b + 7] = yaw_quat(yb)
    w = _qpos_adr(m, 'wx')
    x0[w:w + 6] = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
    x0[_qpos_adr(m, 'jl')] = x0[_qpos_adr(m, 'jr')] = 0.04
    g = x0.copy()
    g[a:a + 3] = pb + np.array([0, 0, 2 * h])
    g[a + 3:a + 7] = yaw_quat(yb + np.pi / 2)
    Q = pose_Q(m, 'cubeA') + pose_Q(m, 'cubeB', w_rot=0.1)
    u_ref = np.concatenate([x0[w:w + 6], [0.04] * (m.nu - 6)])   # one target per jaw actuator
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-1] * (m.nu - 6))
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)


@register
class GripperCubeLift(GripperCubeStack):
  """Diagnostic rung below GripperCubeStack: grasp cube A and lift it 8 cm straight up (same
  x, y, yaw); B stays put.  Same model, contacts and cost weights."""
  HORIZON = 50

  def __init__(self, seed=0):
    super().__init__(seed)
    m = self.model; a = _qpos_adr(m, 'cubeA')
    g = self.x_0.copy(); g[a + 2] += 0.08
    object.__setattr__(self, 'x_goal', g)


@register
class GripperCubeLiftPregrasp(GripperCubeLift):
  """Diagnostic: GripperCubeLift starting in the pre-grasp (wrist at grasp height around cube A, aligned
  with its faces, jaws open 4 cm): only the squeeze and the lift remain."""
  HORIZON = 30

  def __init__(self, seed=0):
    super().__init__(0)   # fixed cube pose: the wrist start (hence the servo reference) must not depend on the seed;
                          # seeds are repeats (GPU nondeterminism only)
    m = self.model; a = _qpos_adr(m, 'cubeA'); w = _qpos_adr(m, 'wx')
    qa = self.x_0[a + 3:a + 7]; ya = 2 * np.arctan2(qa[3], qa[0]); ya -= np.round(ya / (np.pi / 2)) * np.pi / 2
    x0 = self.x_0.copy(); x0[w:w + 6] = [x0[a], x0[a + 1], CUBE_H + 0.035, 0, 0, ya]
    u = self.u_ref.copy(); u[:6] = x0[w:w + 6]
    object.__setattr__(self, 'x_0', x0); object.__setattr__(self, 'u_ref', u)


@register
class GripperCubeLiftGrasped(GripperCubeLiftPregrasp):
  """Diagnostic: as GripperCubeLiftPregrasp but the jaws already rest on the cube faces (0.1 mm in) with
  the grip commanded closed (u_ref grip = 0): only the lift remains."""

  def __init__(self, seed=0):
    super().__init__(seed)
    m = self.model; x0 = self.x_0.copy()
    x0[_qpos_adr(m, 'jl')] = x0[_qpos_adr(m, 'jr')] = CUBE_H - 1e-4
    u = self.u_ref.copy(); u[6:] = 0.0
    object.__setattr__(self, 'x_0', x0); object.__setattr__(self, 'u_ref', u)


def _add_still(task, joint, w_vel, running=False):
  """Weight on the object's (linear + angular) velocity at the goal: 'at the pose and at rest', which rules out
  passing through the goal pose at t = T.  Terminal only by default (Qf and the success metric_Q); running=True
  also puts it in the stage cost Q, which penalizes moving the object at all (first test, 2026-11-23: the
  solver then stopped lifting -- the running velocity cost of an 8 cm lift exceeds the goal's value)."""
  m = task.model; v = _qvel_adr(m, joint)
  Qm = np.array(task.Q, copy=True)
  for i in range(6):
    Qm[v + i, v + i] = w_vel
  object.__setattr__(task, 'metric_Q', Qm)
  object.__setattr__(task, 'Qf', 40 * Qm)
  if running:
    object.__setattr__(task, 'Q', Qm)


def _still_variants(base, name):
  """<name>RunStill: velocity weight in stage + terminal cost (first test); <name>Still: terminal only."""
  class RunStill(base):
    def __init__(self, seed=0):
      super().__init__(seed); _add_still(self, 'cubeA', 1.0, running=True)
  class Still(base):
    def __init__(self, seed=0):
      super().__init__(seed); _add_still(self, 'cubeA', 1.0)
  RunStill.__name__ = RunStill.__qualname__ = name + 'RunStill'
  Still.__name__ = Still.__qualname__ = name + 'Still'
  RunStill.__doc__ = f'{name} + cube velocity weight 1 in Q and Qf (0.1 m/s alone costs 0.01 at the goal).'
  Still.__doc__ = f'{name} + cube velocity weight 1 at the goal only (Qf and success metric).'
  register(RunStill); register(Still)


for _b in (GripperCubeLift, GripperCubeLiftPregrasp, GripperCubeLiftGrasped):
  _still_variants(_b, _b.__name__)


# Grip command range extended past closed (-3 cm): with "closed" on the control bound, the IP's log barrier on
# the bounds pulls an unneeded grip command to the middle of [0, 0.0425] = 0.021 m, i.e. a squeeze of
# kp * 4 mm = 1.6 N per jaw on a 5 cm cube (the slip threshold under acceleration).  With [-0.03, 0.0425] the
# range centre is a firm squeeze (kp * 1.9 cm = 7.6 N).
def _grip_variant(base):
  class G(base):
    GRIP_LO = -0.03
  G.__name__ = G.__qualname__ = base.__name__ + 'G'
  G.__doc__ = f'{base.__name__} with the grip command range extended to [-0.03, 0.0425] (squeeze past closed).'
  return register(G)


for _b in [GripperCubeLift, GripperCubeLiftPregrasp, GripperCubeLiftGrasped, GripperCubeStack]:
  _grip_variant(_b)


# Pad variants (2026-11-23): the grasped lift fails in PLF-V's DDP at a "squirt" cliff of the stiff two-point
# pinch (a slightly larger step ejects the cube at > 5 m/s).  S = softer pad contact (timeconst 0.2 s / 0.4 s:
# ~4x / ~16x lower stiffness than the 0.1 s floor), P = 2 x 2 sphere grid per pad (+-6 mm across the pad width;
# 8 pad spheres, 20 contacts instead of 16).
PAD_VARIANTS = {'S2': dict(PAD_SOLREF=(0.2, 1.0)), 'S4': dict(PAD_SOLREF=(0.4, 1.0)),
                'P': dict(PAD_X=(-0.006, 0.006)), 'PS2': dict(PAD_X=(-0.006, 0.006), PAD_SOLREF=(0.2, 1.0))}


def _pad_variant(base, tag, attrs):
  cls = type(base.__name__ + tag, (base,), dict(attrs))
  cls.__qualname__ = cls.__name__
  cls.__doc__ = f'{base.__name__} with pad variant {tag}: {attrs}.'
  return register(cls)


for _b in [GripperCubeLift, GripperCubeLiftPregrasp, GripperCubeLiftGrasped, GripperCubeStack]:
  for _tag, _attrs in PAD_VARIANTS.items():
    _pad_variant(_b, _tag, _attrs)


def _pregrasp_offset(name, dx, dyaw, dz=0.0):
  """Diagnostic: GripperCubeLiftPregrasp with the wrist start offset by dx (m, along wrist x), dyaw (rad), dz (m).
  How wide is the basin from which PLF-V finds the squeeze-and-lift?"""
  class Off(GripperCubeLiftPregrasp):
    def __init__(self, seed=0):
      super().__init__(seed)
      m = self.model; w = _qpos_adr(m, 'wx'); x0 = self.x_0.copy()
      x0[w] += dx; x0[w + 2] += dz; x0[w + 5] += dyaw
      u = self.u_ref.copy(); u[:6] = x0[w:w + 6]
      object.__setattr__(self, 'x_0', x0); object.__setattr__(self, 'u_ref', u)
  Off.__name__ = Off.__qualname__ = name
  return register(Off)


_pregrasp_offset('GripperCubeLiftPregraspX2', 0.02, 0.0)
_pregrasp_offset('GripperCubeLiftPregraspYaw3', 0.0, 0.3)
_pregrasp_offset('GripperCubeLiftPregraspX2Yaw3', 0.02, 0.3)
_pregrasp_offset('GripperCubeLiftPregraspZ5', 0.0, 0.0, 0.05)


@register
class GripperCubeLiftLong(GripperCubeLift):
  """GripperCubeLift with a 3.5 s horizon (the 2.5 s original leaves only 0.3 s beyond the 2.2 s witness)."""
  HORIZON = 70


@register
class GripperCubeStackLong(GripperCubeStack):
  """GripperCubeStack with a 6 s horizon (the 4.5 s original leaves only 0.2 s beyond the 4.3 s witness)."""
  HORIZON = 120


def problem_size(task):
  """(nx, nw, n_contact_dims): lifted LMS sizes; n_contact_dims = nw - nu (4 per condim-3 contact)."""
  from mujodico.examples.methods.transcription_lifted_multiple_shooting import LiftedMultipleShootingTranscription
  tr = LiftedMultipleShootingTranscription(task)
  return tr.nx, tr.nw, tr.nw - task.nu


# ---------------------------------------------------------------------------------------------
# Task groups in their own modules (each defines Task classes with a `witness(self)` method that
# returns open-loop controls (N, nu), and registers them with bench_tasks.register).
# ---------------------------------------------------------------------------------------------
import importlib, sys as _sys
for _mod in ('tasks_gripper', 'tasks_tools_a', 'tasks_tools_b'):
  try:
    importlib.import_module(_mod)
  except ModuleNotFoundError as _e:
    if _e.name != _mod:
      raise
