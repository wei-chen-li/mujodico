"""SimToolReal-style Allegro tool-use tasks, group B: AllegroBrushSweep, AllegroPenBoard,
AllegroEraserWipe (2026-11-23 bench).

Shared design (all three tasks)
-------------------------------
* Embodiment: `embodiments.allegro_scene` (Allegro on a 6-DoF position-servoed wrist), palm-down
  start (palm quat `quat_mul(axis_quat(x, pi), [0,1,0,1]/sqrt2)`: fingers point +x and flex toward
  -z, thumb opposes from the -x side).  The tool starts lying on the table; the hand must pinch it,
  lift it, reorient it ~90 deg and use it on a target that the hand itself cannot touch.
* Contact filter (contype / conaffinity bits, this module):
    1  hand fingertip capsules + palm capsules (ca 1)  <->  tool grasp geoms (ct 1)
    2  tool support spheres (ct 2)                      <->  table (ca 2)
    8  fingertip spheres added at the 4 tip ends (ca 8) <->  table (ct 8)
   16  tool working geom (ct 16)                        <->  targets: debris / pins (ca 16)
  The palm is excluded from the tool (explicit <exclude>): a palm-down pinch never needs it, and
  the 2 palm capsules would cost 2 contacts.  Palm vs table is replaced by a wrist-height floor
  (lower range of the `wz` joint).  The hand (tips, palm, links) never touches the targets.
* The solver only supports a fixed terminal goal, so the task outcome is recorded physically in the
  final state (debris positions, pin depths); the tool pose has weight 0 in the goal cost.
* The turn cannot be skipped: the targets lie below what the tool's working geometry can reach in
  its lying pose (pen: lying tip centre 16 mm vs 7 mm needed to bottom a pin; brush: lying bristle
  block 26 mm up vs crumb tops at 9 mm; eraser: side faces held 12 mm up by feet vs pins 10 mm proud
  that must go flush).  Minimum turn for the work to be possible at all (geometry, supports resting on
  the table): pen >= ~37 deg (tip-end support sphere), brush ~45 deg, eraser ~45 deg (a felt edge then
  reaches flush and could press a whole 3-pin column); the witnesses turn ~90 deg.
* Every model is seed-independent (seeds only change x_0): simtools.sim_true / the solvers compile
  once per (task class, horizon) and do not refresh the model per seed.
* Witnesses: palm-down pinch (handik IK), lift, roll / turn with the wrist, use.  Grasped tools creep
  in the pinch (rolling about the handle axis: regularized friction under sustained load), so each
  witness aims the tool using its pose in the wrist frame measured on seed 0 under the true law
  (`W_REL`), not the nominal pinch geometry.

True-law pitfalls found while building these (MJX SAP + Hunt-Crossley, smoothing 0):
* Joint `frictionloss` on light DOFs (pins, 7 g effective) is unreliable: the SAP solve reports a
  saturated dry-friction force on pins nobody touches and `_step_given_efc_force` integrates it
  (phantom pin motion, pins jumping past their range).  Pins here use viscous damping instead.
* Box-sphere / sphere-sphere contacts WITH friction between the tool and the pins blew up (NaN) when
  several pins touched the tool at ~zero distance at once (reported friction impulses ~10x outside
  the friction cone).  Tool-pin contacts are therefore condim 1 (frictionless; pins are vertical
  sliders, friction is irrelevant) -- which also cuts their constraint rows from 4 to 1.
* A very light pen (25 g, axial inertia 1.4e-6) exploded under stiff tip-pin contacts; its inertia
  is inflated (see AllegroPenBoard).
"""
import numpy as np
import mujoco
import embodiments as E
from embodiments import fmt
from bench_tasks import register, BenchTask, _qpos_adr, _diag

Q_PALM_DOWN = E.quat_mul(E.axis_quat([1, 0, 0], np.pi), np.array([0, 1, 0, 1]) / np.sqrt(2))
TIP_BODIES = ('ff_tip', 'mf_tip', 'rf_tip', 'th_tip')
TIP_SPHERE_POS = {'ff_tip': (0, 0, 0.029), 'mf_tip': (0, 0, 0.029), 'rf_tip': (0, 0, 0.029),
                  'th_tip': (0, 0, 0.043)}


def allegro_tool_scene(env_world, *, tool_body, wrist_pos=(0, 0, 0), ranges=E.DEFAULT_RANGES,
                       tip_table=True, palm_tool=False, **kw):
  """`embodiments.allegro_scene` + (optionally) one sphere (r 12 mm, massless) at each fingertip end
  that collides with the table only (bit 8), and the palm excluded from `tool_body`."""
  extra = kw.pop('extra_contact', '')
  if not palm_tool:
    extra += f'\n    <exclude body1="palm" body2="{tool_body}"/>'
  xml = E.allegro_scene(env_world, wrist_pos=wrist_pos, palm_quat=Q_PALM_DOWN, ranges=ranges,
                        extra_contact=extra, **kw)
  if tip_table:
    for b in TIP_BODIES:
      tag = f'<body name="{b}" gravcomp="1">'
      assert tag in xml, b
      xml = xml.replace(tag, tag + f'\n<geom name="{b}_sph" type="sphere" size="0.012" pos="{fmt(TIP_SPHERE_POS[b])}" '
                                   f'mass="0" contype="0" conaffinity="8" group="3" rgba="0 1 0 0.3"/>')
  return xml


def allegro_tool_model(env_world, **kw):
  return E.from_xml_in(E.ALLEGRO_DIR, allegro_tool_scene(env_world, **kw))


# ---------------------------------------------------------------------------------------------
# Helpers: rotations, wrist poses, palm-down pinch IK
# ---------------------------------------------------------------------------------------------
def rx(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrist_R(e):
  """Wrist orientation for hinge angles (wrx, wry, wrz): R = Rx Ry Rz (joints in series)."""
  return rx(e[0]) @ ry(e[1]) @ rz(e[2])


def euler_xyz(R):
  """Inverse of wrist_R (b in (-pi/2, pi/2))."""
  b = np.arcsin(np.clip(R[0, 2], -1, 1))
  a = np.arctan2(-R[1, 2], R[2, 2])
  c = np.arctan2(-R[0, 1], R[0, 0])
  return np.array([a, b, c])


def euler_xyz_near(R, ref):
  """Wrist angles for R on the branch closest to `ref` (the two solutions of R = Rx Ry Rz are
  (a, b, c) and (a + pi, pi - b, c + pi))."""
  a, b, c = euler_xyz(R)
  w = lambda v: (v + np.pi) % (2 * np.pi) - np.pi
  s1 = np.array([a, b, c]); s2 = np.array([w(a + np.pi), np.pi - b, w(c + np.pi)])
  ref = np.asarray(ref, float)
  return s1 if np.sum((s1 - ref) ** 2) <= np.sum((s2 - ref) ** 2) else s2


def quat_R(q):
  R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def R_quat(R):
  q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


def _align(a, b):
  """Smallest rotation taking unit vector a to unit vector b."""
  a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
  v = np.cross(a, b); c = float(a @ b)
  if np.linalg.norm(v) < 1e-12:
    return np.eye(3)
  K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + K + K @ K / (1 + c)


def wrist_for_object(p_obj, R_obj, rel, ref=None):
  """Wrist (x, y, z, wrx, wry, wrz) that puts the grasped object at (p_obj, R_obj), given the
  object pose in the wrist frame rel = (p_ow, R_ow); `ref` picks the wrist-angle branch."""
  p_ow, R_ow = rel
  Rw = R_obj @ R_ow.T
  e = euler_xyz(Rw) if ref is None else euler_xyz_near(Rw, ref)
  return np.concatenate([p_obj - Rw @ p_ow, e])


def object_in_wrist(x, m, obj_joint):
  """(p_ow, R_ow) of free joint `obj_joint` relative to the wrist in state x."""
  w = _qpos_adr(m, 'wx'); a = _qpos_adr(m, obj_joint)
  Rw = wrist_R(x[w + 3:w + 6]); pw = x[w:w + 3]
  return Rw.T @ (x[a:a + 3] - pw), Rw.T @ quat_R(x[a + 3:a + 7])


def pinch_fingers(m, rel_axis_centre, radius, spread=0.04, dz=0.004, inward=0.0, thumb_base=1.4):
  """Finger joint targets (16,) of the palm-down pinch of a cylinder lying along the wrist-frame y
  axis (wrist at zero angles), centre `rel_axis_centre` (wrist frame), radius `radius`: three finger
  tips on its +x side (y = +spread, 0, -spread), thumb on its -x side, at axis height + dz.
  inward > 0 squeezes (targets past the surface), < 0 opens."""
  import handik as H
  W = _qpos_adr(m, 'wx')
  q = np.zeros(m.nq); q[W + 6 + 12] = thumb_base
  for j in range(m.njnt):     # free joints: identity quats (far away, irrelevant for IK)
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      q[m.jnt_qposadr[j] + 3] = 1; q[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = [0, 0, 5]
  c = np.asarray(rel_axis_centre, float)
  s = radius + 0.012 - 0.002 - inward
  tg = {H.tip_geom(m, 'ff'): c + [s, spread, dz], H.tip_geom(m, 'mf'): c + [s, 0, dz],
        H.tip_geom(m, 'rf'): c + [s, -spread, dz], H.tip_geom(m, 'th'): c + [-s, 0, dz]}
  qs, e = H.ik(m, q, tg, sum(H.FINGER_JOINTS.values(), []), iters=3000)
  return qs[W + 6:W + 22], e


def finger_targets(m, qf):
  names = [m.actuator(i).name for i in range(m.nu)]
  fa = [n for n in names if not n.startswith('a_')]
  return {n: float(v) for n, v in zip(fa, qf)}


def wrist_targets(w):
  return {f'a_{j}': float(v) for j, v in zip(E.WRIST_JOINTS, w)}


TABLE_B = """    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" rgba="0.55 0.45 0.35 1"
          contype="8" conaffinity="2" friction="0.6"/>"""
GRASP_REL = np.array([0.01, 0.013, -0.104])   # tool axis centre in the wrist frame at the pinch
WRIST_FLOOR = 0.015     # wz lower limit: palm-down, the palm capsules' lowest point is 11 mm below the wrist


def _pins_xml(prefix, pts, r_head, h_head, depth, damping, rgba='0.1 0.3 0.9 1'):
  """Push-pins: vertical slide joint (range [-depth, +2 mm]), gravity-compensated and viscously damped
  (no restoring force, so a pin stays wherever it is pushed), sphere head (r_head, centre h_head above
  the board) that only the tool's working geom touches (ca 16, condim 1).  No joint `frictionloss`
  (unreliable on light DOFs under the true law, see the module docstring)."""
  out = []
  for i, (x, y) in enumerate(pts):
    out.append(f"""    <body name="{prefix}{i}" pos="{x:.4f} {y:.4f} 0" gravcomp="1">
      <joint name="{prefix}{i}" type="slide" axis="0 0 1" range="{-depth} 0.002" damping="{damping}" armature="0.002"/>
      <geom name="{prefix}{i}_head" type="sphere" size="{r_head}" pos="0 0 {h_head}" mass="0.005" contype="0" conaffinity="16" rgba="{rgba}" condim="1"/>
      <geom name="{prefix}{i}_stem" type="cylinder" size="0.002 {h_head / 2}" pos="0 0 {h_head / 2}" mass="0" contype="0" conaffinity="0" rgba="0.8 0.8 0.8 1"/>
    </body>""")
  return '\n'.join(out)


class _AllegroToolTask(BenchTask):
  DT = 0.04
  FINGER_KP = 1.0
  FINGER_FORCE = 2.0

  def finish(self, m, x0, g, Q, horizon, metric_Q=None):
    W = _qpos_adr(m, 'wx')
    u_ref = x0[W:W + 22].copy()
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-2] * 16)
    BenchTask.__init__(self, model=m, horizon=horizon, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q, u_ref=u_ref, R=R,
                       contact_smoothing=0.002)
    if metric_Q is not None:
      object.__setattr__(self, 'metric_Q', metric_Q)


# =============================================================================================
# AllegroPenBoard
# =============================================================================================
PEN_R = 0.016          # marker barrel radius (thick marker, so the tips reach below its equator)
PEN_HALF = 0.065       # barrel half length (capsule segment)
PEN_INERTIA = (1e-4, 2e-5)   # transverse, axial kg m^2 (inflated, see docstring); mass 25 g
PEN_TIP = (0.08, 0.007)   # tip sphere: distance from the centre along the pen axis, radius
PIN_SPACING = 0.02
PIN_DEPTH = 0.012
PIN_DAMPING = 1.0      # N s/m: pressing a pin at 5 cm/s takes 0.05 N
PIN_HEAD = (0.006, PIN_DEPTH - 0.006)   # radius, centre height (unpressed): pressed = head top flush with the table
L_LETTER = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (2.0, 0.0)]   # 'L', units of PIN_SPACING


@register
class AllegroPenBoard(_AllegroToolTask):
  """SimToolReal 'pen / marker writing': reorient a pen and write a letter on a whiteboard.

  Modelled: a thick marker (barrel capsule r 16 mm, 13 cm + caps; a tip sphere r 7 mm on its axis
  8 cm from the centre, i.e. 1.5 cm beyond the barrel end; 25 g) lying flat on the table along ~world
  y, tip toward -y.  A whiteboard area on the table holds 5 'ink pins' laid out as the letter 'L'
  (2 cm pitch): each pin is a vertical slide joint (range [-12 mm, +2 mm], gravity-compensated,
  damping 1 N s/m, no restoring force: it stays wherever it is pushed) with a sphere head (r 6 mm)
  standing 12 mm proud; pressed = head top flush with the board.  Writing = pressing every pin of
  the letter to its stop with the pen tip: the hand must pinch the pen, lift it, turn it upright
  (~90 deg about the pinch axis; >= ~37 deg is geometrically necessary, since the barrel-end support
  sphere keeps a flatter tip too high) and dab the letter.
  Simplifications: the ink stroke is discretised into pins; the pen is a thick marker (r 16 mm) so the
  palm-down Allegro pinch can reach below its equator without the fingertips hitting the table (a
  r 12 mm pen squirted out of the pinch); pen inertia is inflated to (1e-4, 2e-5 axial) kg m^2
  (real ~4e-5 / 3e-6) because the true law blew up with the real values; only the pen TIP touches
  the pins (frictionless; hand, barrel and caps excluded) and the tip does not touch the table
  (pins bottom out on their stops); the barrel rests on two support spheres (r 16 mm at the barrel
  ends) that touch the table only.
  Contacts (15; 45 constraint rows): 4 fingertip capsules vs barrel (capsule-capsule); 2 pen support
  spheres vs table; 4 fingertip spheres vs table (palm vs table -> wrist floor wz >= 0.015); pen tip
  vs 5 pin heads (condim 1).
  Cost: 300 * sum_i (q_pin_i + 12 mm)^2 (pen pose weight 0); Qf = 40 Q.  Success < 0.01: e.g. all
  pins within 2.5 mm of the stop, or one pin short by <= 5.7 mm with the rest bottomed out; an
  untouched pin alone costs 0.043 (fail).  Seeds randomize the pen's (x +-1.5 cm, y +-0.5 cm,
  yaw +-0.15) start.  Witness: 5.2 s (130 steps)."""
  HORIZON = 150
  W_PIN = 300.0
  GRASP_DZ = -0.003    # pinch 3 mm below the barrel equator (squeeze pulls the pen toward the palm)
  SQUEEZE = 0.02

  def __init__(self, seed=0):
    rng = np.random.default_rng(7100 + seed)
    pp = np.array([0.04 + rng.uniform(-0.015, 0.015), -0.10 + rng.uniform(-0.005, 0.005), PEN_R - 1e-4])
    yaw = rng.uniform(-0.15, 0.15)
    self.pen_p0, self.pen_yaw = pp, yaw
    self.pins = np.array([[0.0 + PIN_SPACING * a, 0.06 + PIN_SPACING * b] for a, b in L_LETTER])
    rt = PEN_TIP[1]
    env = TABLE_B + f"""
    <geom name="board" type="box" size="0.06 0.06 0.001" pos="0.02 0.08 0.001" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    <body name="pen" pos="0 0 0.3">
      <freejoint name="pen"/>
      <inertial pos="0 0 0" mass="0.025" diaginertia="{PEN_INERTIA[0]} {PEN_INERTIA[1]} {PEN_INERTIA[0]}"/>
      <geom name="pen_barrel" type="capsule" size="{PEN_R}" fromto="0 {-PEN_HALF} 0 0 {PEN_HALF} 0" contype="1" conaffinity="0" friction="1" rgba="0.9 0.9 0.2 1"/>
      <geom name="pen_sup0" type="sphere" size="{PEN_R}" pos="0 {-PEN_HALF} 0" mass="0" contype="2" conaffinity="0" group="3"/>
      <geom name="pen_sup1" type="sphere" size="{PEN_R}" pos="0 {PEN_HALF} 0" mass="0" contype="2" conaffinity="0" group="3"/>
      <geom name="pen_tip" type="sphere" size="{rt}" pos="0 {-PEN_TIP[0]} 0" contype="16" conaffinity="0" rgba="0.1 0.1 0.1 1" condim="1"/>
    </body>
{_pins_xml('pin', self.pins, PIN_HEAD[0], PIN_HEAD[1], PIN_DEPTH, PIN_DAMPING)}"""
    ranges = ((-0.35, 0.35), (-0.35, 0.35), (WRIST_FLOOR, 0.4), (-2.0, 2.0), (-1.6, 1.6), (-1.6, 1.6))
    m = allegro_tool_model(env, tool_body='pen', ranges=ranges, timestep=self.DT, finger_kp=self.FINGER_KP,
                           finger_force=self.FINGER_FORCE)
    self.qf_open, _ = pinch_fingers(m, GRASP_REL, PEN_R, inward=-0.03, dz=self.GRASP_DZ)
    self.qf_squeeze, _ = pinch_fingers(m, GRASP_REL, PEN_R, inward=self.SQUEEZE, dz=self.GRASP_DZ)
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    a = _qpos_adr(m, 'pen'); W = _qpos_adr(m, 'wx')
    x0[a:a + 3] = pp; x0[a + 3:a + 7] = R_quat(rz(yaw))
    # wrist above the pen, at the pen yaw, fingers pre-shaped open
    wg = wrist_for_object(pp, rz(yaw), (GRASP_REL, np.eye(3)))
    self.w_grasp = wg
    x0[W:W + 6] = wg + np.array([0, 0, 0.06, 0, 0, 0])
    x0[W + 6:W + 22] = self.qf_open
    g = x0.copy()
    ent = {}
    for i in range(len(self.pins)):
      g[_qpos_adr(m, f'pin{i}')] = -PIN_DEPTH
      ent[_qpos_adr(m, f'pin{i}')] = self.W_PIN
    Q = _diag(nx, ent)
    self.finish(m, x0, g, Q, self.HORIZON)

  # witness ------------------------------------------------------------------------------------
  # pen centre and axis in the wrist frame once turned upright, measured on seed 0 under the true law
  # (the pen rolls ~40 deg about its own axis and shifts ~1 cm in the pinch; only the axis and the
  # tip offset matter for the dabs)
  W_REL = ([0.00688, 0.01796, -0.12131], [0.14146, 0.98989, -0.01066])

  def witness(self, rel=None, t_pin=0.6, z_press=PEN_TIP[1], hover=0.015, t_rot=2.2, fr=(0.25, 0.6, 0.75), kf_only=False):
    from simtools import keyframes
    m = self.model
    rel = rel or self.W_REL
    p_ow = np.asarray(rel[0], float)
    ax = np.asarray(rel[1], float)             # pen axis (tip -> back) in the wrist frame
    if ax.ndim == 2:                           # (p_ow, R_ow) as measured
      ax = ax[:, 1]
    ax = ax / np.linalg.norm(ax)
    wg = self.w_grasp
    p_tw = p_ow - PEN_TIP[0] * ax              # tip in the wrist frame
    Rw_up = rx(np.pi / 2) @ _align(ax, np.array([0, 1, 0]))   # wrist orientation: pen vertical, tip down
    def w_tip(p_tip):                          # wrist pose putting the pen tip at p_tip
      return np.concatenate([p_tip - Rw_up @ p_tw, euler_xyz(Rw_up)])
    z_hov = PIN_HEAD[1] + PIN_HEAD[0] + PEN_TIP[1] + hover
    kf = [(0.5, wrist_targets(wg)), (0.6, {}), (0.9, finger_targets(m, self.qf_squeeze)), (1.0, {}),
          (1.4, wrist_targets(wg + [0, 0, 0.08, 0, 0, 0]))]
    p0 = self.pins[0]
    kf.append((t_rot, wrist_targets(w_tip(np.array([p0[0], p0[1], z_hov + 0.02])))))
    t = t_rot
    for p in self.pins:      # dab: move over the pin, descend gently onto the stop, hold, rise
      kf.append((t + fr[0] * t_pin, wrist_targets(w_tip(np.array([p[0], p[1], z_hov])))))
      kf.append((t + fr[1] * t_pin, wrist_targets(w_tip(np.array([p[0], p[1], z_press])))))
      kf.append((t + fr[2] * t_pin, {}))
      kf.append((t + t_pin, wrist_targets(w_tip(np.array([p[0], p[1], z_hov])))))
      t += t_pin
    if kf_only:
      return kf
    return keyframes(self, kf)


# =============================================================================================
# AllegroBrushSweep
# =============================================================================================
BR_R = 0.015            # handle radius
BR_HALF = 0.065         # handle half length (body y in [-BR_HALF, BR_HALF])
BR_HEAD = (0.03, 0.03, 0.005)  # bristle block half sizes: x (bristle direction), y (width), z (thickness)
BR_HEAD_Z = 0.016       # head block offset from the handle axis along body z (lying: up, off the table)
BR_GRASP_Y = -0.02      # pinch centre on the handle (body y)
BR_MASS = (0.02, 0.008)  # handle, bristle block (kg): light (regularized-friction creep in the pinch grows with load)
DEB_R = 0.0045          # debris ("crumb") sphere radius (planar, centre at z = DEB_R)
DEB_DAMPING = 0.15      # N s/m on the debris slides (10 g: 67 ms velocity time constant)
PAN_X = 0.165           # dustpan back wall (inner face, world x)
DEB_GOAL_X = PAN_X - DEB_R   # debris targets: against the pan's back wall
DEB_GOAL_Y = (0.075, 0.115)


@register
class AllegroBrushSweep(_AllegroToolTask):
  """SimToolReal 'brush sweeping': grasp a hand brush, turn its bristles down, sweep debris into a pan.

  Modelled: a brush lying on the table -- handle capsule (r 15 mm, 13 cm + caps, 20 g, along ~world y)
  and, past its +y end, a bristle block (box 6 x 6 x 1 cm, 8 g) offset 16 mm from the handle axis,
  whose bristles point SIDEWAYS (-x, horizontal).  The hand must pinch the handle palm-down, lift it
  and roll it ~90 deg about the handle axis so the bristles point down (the wrist rolls 60 deg, the
  handle rolls another ~30 deg inside the pinch, thumb ending under the handle), then sweep toward +x.
  Two crumbs (spheres r 4.5 mm, 10 g) are planar bodies (x, y slide joints, damping 0.15 N s/m, fixed
  height, as in the repo's PushT task); the dustpan is a floor region x in [9.5, 16.5] cm with its
  back wall at x = 16.5 cm (a box that stops crumbs).  Lying, the bristle block is 26 mm off the table
  (crumb tops at 9 mm), so the brush must be rolled >= ~45 deg before it can touch them.
  Simplifications: crumbs are planar (no z / rotation) and only the bristle block touches them (hand,
  handle and supports excluded); the pan is its back wall only (the brush passes through it); the
  brush rests on 3 support spheres (both handle ends; a r 12.5 mm foot under the head, which stays
  6 mm short of the bristle tips once they point down) that touch the table only -- the bristle block
  itself never touches the table; light brush (28 g): the pinch creeps under load; fingertips touch
  the handle only.
  Contacts (15; 60 constraint rows): 4 fingertip capsules vs handle (capsule-capsule); 3 support
  spheres vs table; 4 fingertip spheres vs table (palm vs table -> wrist floor wz >= 0.015); bristle
  block vs 2 crumbs (box-sphere, friction 0.5); pan wall vs 2 crumbs.
  Cost: 25 * sum_i |crumb_i(x, y) - target_i|^2, targets against the pan's back wall (x = 16.05 cm,
  y = 7.5 / 11.5 cm), brush pose weight 0; Qf = 40 Q.  Success < 0.01: e.g. both crumbs within 1.4 cm
  of their spots, or one exact and the other within 2 cm (the wall makes x easy; y is the hard part).
  Seeds randomize the brush (x, y +-1 cm, yaw +-0.12) and the crumb starts (+-0.6 cm).
  Witness: 4.5 s (113 steps)."""
  HORIZON = 130
  W_DEB = 25.0
  GRASP_DZ = -0.003
  SQUEEZE = 0.02

  def __init__(self, seed=0):
    rng = np.random.default_rng(7300 + seed)
    pb = np.array([-0.07 + rng.uniform(-0.01, 0.01), -0.08 + rng.uniform(-0.01, 0.01), BR_R - 1e-4])
    yaw = rng.uniform(-0.12, 0.12)
    self.deb0 = np.array([[0.0, DEB_GOAL_Y[0]], [-0.015, DEB_GOAL_Y[1]]]) + rng.uniform(-0.006, 0.006, (2, 2))
    hx, hy, hz = BR_HEAD
    hc = (-hx, BR_HALF + hy, BR_HEAD_Z)    # head block centre (body): bristles toward body -x
    # supports (table only): both handle ends (on the axis) and a foot (sphere r 12.5 mm) under the head,
    # which holds the lying head 2.6 cm off the table (above the debris) and stays 6 mm short of the
    # bristle tips when they point down
    rf = 0.0125
    sup = [(0.0, -BR_HALF, 0.0, BR_R), (0.0, BR_HALF, 0.0, BR_R), (-2 * hx + 0.006 + rf, BR_HALF + hy, -BR_R + rf, rf)]
    sup_xml = '\n'.join(f'      <geom name="br_sup{i}" type="sphere" size="{r}" pos="{x} {y} {z}" mass="0" contype="2" conaffinity="0" group="3"/>'
                        for i, (x, y, z, r) in enumerate(sup))
    deb_xml = '\n'.join(f"""    <body name="deb{i}" pos="0 0 {DEB_R}">
      <joint name="deb{i}_x" type="slide" axis="1 0 0" damping="{DEB_DAMPING}"/>
      <joint name="deb{i}_y" type="slide" axis="0 1 0" damping="{DEB_DAMPING}"/>
      <geom name="deb{i}" type="sphere" size="{DEB_R}" mass="0.01" contype="0" conaffinity="16" friction="0.5" rgba="0.5 0.35 0.2 1"/>
    </body>""" for i in range(2))
    env = TABLE_B + f"""
    <geom name="pan" type="box" size="0.035 0.07 0.001" pos="{PAN_X - 0.035} 0.095 0.001" rgba="0.2 0.6 0.3 1" contype="0" conaffinity="0"/>
    <geom name="pan_wall" type="box" size="0.005 0.07 0.02" pos="{PAN_X + 0.005} 0.095 0.02" rgba="0.2 0.6 0.3 1" contype="16" conaffinity="0"/>
    <body name="brush" pos="0 0 0.3">
      <freejoint name="brush"/>
      <geom name="br_handle" type="capsule" size="{BR_R}" fromto="0 {-BR_HALF} 0 0 {BR_HALF} 0" mass="{BR_MASS[0]}" contype="1" conaffinity="0" friction="1" rgba="0.6 0.4 0.2 1"/>
      <geom name="br_head" type="box" size="{hx} {hy} {hz}" pos="{fmt(hc)}" mass="{BR_MASS[1]}" contype="16" conaffinity="0" friction="0.5" rgba="0.9 0.85 0.6 1"/>
{sup_xml}
    </body>
{deb_xml}"""
    ranges = ((-0.35, 0.35), (-0.35, 0.35), (WRIST_FLOOR, 0.4), (-1.6, 1.6), (-2.0, 1.6), (-1.6, 1.6))
    m = allegro_tool_model(env, tool_body='brush', ranges=ranges, timestep=self.DT, finger_kp=self.FINGER_KP,
                           finger_force=self.FINGER_FORCE)
    self.qf_open, _ = pinch_fingers(m, GRASP_REL, BR_R, inward=-0.03, dz=self.GRASP_DZ)
    self.qf_squeeze, _ = pinch_fingers(m, GRASP_REL, BR_R, inward=self.SQUEEZE, dz=self.GRASP_DZ)
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    a = _qpos_adr(m, 'brush'); W = _qpos_adr(m, 'wx')
    x0[a:a + 3] = pb; x0[a + 3:a + 7] = R_quat(rz(yaw))
    self.rel0 = (GRASP_REL - np.array([0, BR_GRASP_Y, 0]), np.eye(3))   # brush in the wrist frame at the pinch
    wg = wrist_for_object(pb, rz(yaw), self.rel0)
    self.w_grasp = wg
    x0[W:W + 6] = wg + np.array([0, 0, 0.06, 0, 0, 0])
    x0[W + 6:W + 22] = self.qf_open
    for i in range(2):
      x0[_qpos_adr(m, f'deb{i}_x')] = self.deb0[i, 0]; x0[_qpos_adr(m, f'deb{i}_y')] = self.deb0[i, 1]
    g = x0.copy(); ent = {}
    for i in range(2):
      ax, ay = _qpos_adr(m, f'deb{i}_x'), _qpos_adr(m, f'deb{i}_y')
      g[ax] = DEB_GOAL_X; g[ay] = DEB_GOAL_Y[i]
      ent[ax] = ent[ay] = self.W_DEB
    Q = _diag(nx, ent)
    self.finish(m, x0, g, Q, self.HORIZON)

  # witness ------------------------------------------------------------------------------------
  # brush pose in the wrist frame while sweeping, measured on seed 0 under the true law (None ->
  # nominal pinch).  The wrist positions aim the head's leading bristle edge using this pose.
  W_REL = ([0.00745, 0.0372, -0.11818], [[0.84251, 0.12115, -0.52488], [-0.08797, 0.99224, 0.08782], [0.53145, -0.02781, 0.84663]])

  def witness(self, rel=None, z_lift=0.11, t_roll=2.2, z_sweep=0.002, wry=-1.05, gap=-0.02, t_sweep=1.2, back=0.045, level=False, kf_only=False):
    from simtools import keyframes
    m = self.model
    rel = rel or self.W_REL or self.rel0
    p_ow, R_ow = np.asarray(rel[0], float), np.asarray(rel[1], float)
    wg = self.w_grasp
    hx, hy, hz = BR_HEAD
    if level and rel is not self.rel0:     # wrist orientation that puts the bristles straight down, handle along world y
      e_sw = euler_xyz_near(ry(-np.pi / 2) @ R_ow.T, (0, wry, 0))
    else:                                  # nominal: roll the wrist about world y
      e_sw = np.array([0.0, wry, 0.0])
    Rw = wrist_R(e_sw)
    p_ref = p_ow + R_ow @ np.array([-2 * hx, BR_HALF + hy, BR_HEAD_Z - hz])   # leading bristle edge (wrist frame)
    def w_head(x_face, y_c, z_bot):        # wrist pose putting the leading bristle edge at (x_face, y_c, z_bot)
      return np.concatenate([np.array([x_face, y_c, z_bot]) - Rw @ p_ref, e_sw])
    yc = 0.5 * (DEB_GOAL_Y[0] + DEB_GOAL_Y[1])
    xs = min(self.deb0[:, 0]) - DEB_R - back     # sweep start, well behind the debris
    x_end = DEB_GOAL_X - DEB_R - gap
    kf = [(0.5, wrist_targets(wg)), (0.6, {}), (0.9, finger_targets(m, self.qf_squeeze)), (1.0, {}),
          (1.5, wrist_targets(wg + [0, 0, z_lift, 0, 0, 0]))]
    kf.append((t_roll, wrist_targets(w_head(xs, yc, 0.06))))
    kf.append((t_roll + 0.5, wrist_targets(w_head(xs, yc, z_sweep))))
    t1 = t_roll + 0.5 + t_sweep
    kf.append((t1, wrist_targets(w_head(x_end, yc, z_sweep))))
    kf.append((t1 + 0.3, {}))
    kf.append((t1 + 0.6, wrist_targets(w_head(x_end - 0.01, yc, z_sweep + 0.01))))
    if kf_only:
      return kf
    return keyframes(self, kf)


# =============================================================================================
# AllegroEraserWipe
# =============================================================================================
ER_HR = 0.0125          # handle radius
ER_HALF = 0.037         # handle half length
ER_BLOCK = (0.01, 0.025, 0.0125)   # felt block half sizes: x (thickness, felt normal = body -x), y (length), z (width)
ER_GAP = 0.026          # handle surface to block back face (finger / thumb clearance)
ER_BX = -(ER_HR + ER_GAP + ER_BLOCK[0])   # block centre (body x)
ER_MASS = (0.008, 0.015)   # handle, block (kg)
ER_FOOT = 0.012         # foot height below the side faces
EP_DEPTH = 0.01         # ink pins stand 10 mm proud of the board; flush = head top at the board (z = 0)
EP_HEAD = 0.006
EP_PITCH = (0.03, 0.022)   # grid pitch x (2 columns), y (3 rows)
EP_CENTRE = (0.0, 0.07)


@register
class AllegroEraserWipe(_AllegroToolTask):
  """SimToolReal 'eraser wiping': grasp an eraser, turn its felt face down, wipe the ink off a board.

  Modelled: a handled whiteboard eraser -- felt block 5 x 2.5 x 2 cm (15 g) with a handle bar (capsule
  r 12.5 mm, 7.4 cm + caps, 8 g) parallel to its length, 2.6 cm finger clearance between them (posts
  visual only) -- lying on its SIDE on three 12 mm feet: the felt face is vertical, facing -x.  The hand
  must pinch the handle palm-down, lift it and roll it ~90 deg about the handle axis (wrist 60 deg +
  ~35 deg roll inside the pinch; >= ~45 deg is geometrically necessary) so the felt faces down, then
  press the ink off.  'Ink' = a 2 x 3 grid
  of push-pins (pitch 3 cm in x, 2.2 cm in y) standing 10 mm proud of the board; each pin is a vertical
  slide joint (range [-10 mm, +2 mm], gravity-compensated, damping 1 N s/m, no restoring force) with a
  sphere head (r 6 mm).  Wiped = every pin flush (at its stop).  The felt face (2.5 cm across) covers
  one column of 3 pins at a time, so at least two placements (or a wipe across) are needed.
  Simplifications: only the felt block touches the pins (frictionless; hand and handle excluded); the
  block never touches the table (pins bottom out on their stops); the eraser rests on 3 feet (spheres
  r 6 mm: handle middle, 2 under the block, table only) that hold its side faces 12 mm off the table
  so its sides cannot squash the pins; the middle finger is excluded from the eraser (3-contact pinch:
  index + ring vs thumb, finger spread 2.5 cm) and fingertips touch the handle only.
  Contacts (16; 46 constraint rows): 3 fingertip capsules (ff, rf, th) vs handle; 3 feet vs table;
  4 fingertip spheres vs table (palm vs table -> wrist floor wz >= 0.015); felt block vs 6 pin heads
  (box-sphere, condim 1).
  Cost: 300 * sum_i (q_pin_i + 10 mm)^2 (eraser pose weight 0); Qf = 40 Q.  Success < 0.01: e.g. all
  pins within 2.3 mm of flush; one untouched pin alone costs 0.03 (fail).  Seeds randomize the
  eraser's (x, y +-1 cm, yaw +-0.12) start.  Witness: 4.0 s (100 steps)."""
  HORIZON = 120
  W_PIN = 300.0
  GRASP_DZ = -0.003
  SQUEEZE = 0.02
  SPREAD = 0.025

  def __init__(self, seed=0):
    rng = np.random.default_rng(7500 + seed)
    pe = np.array([-0.07 + rng.uniform(-0.01, 0.01), -0.07 + rng.uniform(-0.01, 0.01), ER_BLOCK[2] + ER_FOOT - 1e-4])
    yaw = rng.uniform(-0.12, 0.12)
    self.pins = np.array([[EP_CENTRE[0] + EP_PITCH[0] * (i - 0.5), EP_CENTRE[1] + EP_PITCH[1] * (j - 1)]
                          for i in range(2) for j in range(3)])
    bx, by, bz = ER_BLOCK
    # three feet (spheres r 6 mm, table only) on the side the eraser lies on: they hold its side faces
    # 12 mm off the table, above the 10 mm proud pins, so only the felt face can push pins flush; with
    # the felt face down they sit >= 8 mm above it
    rf = 0.006
    zf = -(bz + ER_FOOT) + rf
    sup = [(0.0, 0.0, zf, rf), (ER_BX + 0.004, -by + rf, zf, rf), (ER_BX + 0.004, by - rf, zf, rf)]
    sup_xml = '\n'.join(f'      <geom name="er_sup{i}" type="sphere" size="{r}" pos="{x} {y} {z}" mass="0" contype="2" conaffinity="0" group="3"/>'
                        for i, (x, y, z, r) in enumerate(sup))
    env = TABLE_B + f"""
    <geom name="board" type="box" size="0.05 0.05 0.001" pos="{EP_CENTRE[0]} {EP_CENTRE[1]} 0.001" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    <body name="eraser" pos="0 0 0.3">
      <freejoint name="eraser"/>
      <geom name="er_handle" type="capsule" size="{ER_HR}" fromto="0 {-ER_HALF} 0 0 {ER_HALF} 0" mass="{ER_MASS[0]}" contype="1" conaffinity="0" friction="1" rgba="0.2 0.2 0.25 1"/>
      <geom name="er_block" type="box" size="{bx} {by} {bz}" pos="{ER_BX} 0 0" mass="{ER_MASS[1]}" contype="16" conaffinity="0" friction="0.6" rgba="0.3 0.3 0.8 1" condim="1"/>
      <geom name="er_post0" type="box" size="{(ER_GAP + ER_HR) / 2} 0.003 0.006" pos="{-(ER_GAP + ER_HR) / 2} {-by + 0.003} 0" mass="0" contype="0" conaffinity="0" rgba="0.2 0.2 0.25 1"/>
      <geom name="er_post1" type="box" size="{(ER_GAP + ER_HR) / 2} 0.003 0.006" pos="{-(ER_GAP + ER_HR) / 2} {by - 0.003} 0" mass="0" contype="0" conaffinity="0" rgba="0.2 0.2 0.25 1"/>
{sup_xml}
    </body>
{_pins_xml('ink', self.pins, EP_HEAD, EP_DEPTH - EP_HEAD, EP_DEPTH, PIN_DAMPING, rgba='0.1 0.1 0.1 1')}"""
    ranges = ((-0.35, 0.35), (-0.35, 0.35), (WRIST_FLOOR, 0.4), (-1.6, 1.6), (-2.0, 1.6), (-1.6, 1.6))
    m = allegro_tool_model(env, tool_body='eraser', ranges=ranges, timestep=self.DT, finger_kp=self.FINGER_KP,
                           finger_force=self.FINGER_FORCE, extra_contact='\n    <exclude body1="mf_tip" body2="eraser"/>')
    self.qf_open, _ = pinch_fingers(m, GRASP_REL, ER_HR, inward=-0.03, dz=self.GRASP_DZ, spread=self.SPREAD)
    self.qf_squeeze, _ = pinch_fingers(m, GRASP_REL, ER_HR, inward=self.SQUEEZE, dz=self.GRASP_DZ, spread=self.SPREAD)
    self.qf_squeeze[4:8] = self.qf_open[4:8]     # middle finger stays open (excluded from the eraser)
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    a = _qpos_adr(m, 'eraser'); W = _qpos_adr(m, 'wx')
    x0[a:a + 3] = pe; x0[a + 3:a + 7] = R_quat(rz(yaw))
    self.rel0 = (GRASP_REL, np.eye(3))
    wg = wrist_for_object(pe, rz(yaw), self.rel0)
    self.w_grasp = wg
    x0[W:W + 6] = wg + np.array([0, 0, 0.06, 0, 0, 0])
    x0[W + 6:W + 22] = self.qf_open
    g = x0.copy(); ent = {}
    for i in range(len(self.pins)):
      g[_qpos_adr(m, f'ink{i}')] = -EP_DEPTH
      ent[_qpos_adr(m, f'ink{i}')] = self.W_PIN
    Q = _diag(nx, ent)
    self.finish(m, x0, g, Q, self.HORIZON)

  # witness ------------------------------------------------------------------------------------
  # eraser pose in the wrist frame once rolled, measured on seed 0 under the true law (the handle rolls
  # ~35 deg inside the pinch, helping the wrist's 60 deg roll); None -> nominal pinch
  W_REL = ([0.00754, 0.02059, -0.10743], [[0.78959, 0.18854, -0.58395], [-0.06314, 0.97154, 0.2283], [0.61038, -0.14339, 0.77902]])

  def witness(self, rel=None, z_lift=0.10, t_roll=2.2, wry=-1.05, z_press=0.0, hover=0.015, t_col=0.9, level=True, kf_only=False):
    from simtools import keyframes
    m = self.model
    rel = rel or self.W_REL or self.rel0
    p_ow, R_ow = np.asarray(rel[0], float), np.asarray(rel[1], float)
    wg = self.w_grasp
    if level and rel is not self.rel0:     # wrist orientation that puts the felt face level, long axis along world y
      Rw = ry(-np.pi / 2) @ R_ow.T
      e_r = euler_xyz_near(Rw, (0, wry, 0))
    else:
      e_r = np.array([0.0, wry, 0.0])
    Rw = wrist_R(e_r)
    p_ref = p_ow + R_ow @ np.array([ER_BX - ER_BLOCK[0], 0, 0])   # felt face centre (wrist frame)
    def w_face(x, y, z):
      return np.concatenate([np.array([x, y, z]) - Rw @ p_ref, e_r])
    cols = sorted(set(np.round(self.pins[:, 0], 6)))
    yc = EP_CENTRE[1]
    kf = [(0.5, wrist_targets(wg)), (0.6, {}), (0.9, finger_targets(m, self.qf_squeeze)), (1.0, {}),
          (1.5, wrist_targets(wg + [0, 0, z_lift, 0, 0, 0]))]
    kf.append((t_roll, wrist_targets(w_face(cols[0], yc, EP_DEPTH + hover + 0.02))))
    t = t_roll
    for xc in cols:
      kf.append((t + 0.3 * t_col, wrist_targets(w_face(xc, yc, EP_DEPTH + hover))))
      kf.append((t + 0.6 * t_col, wrist_targets(w_face(xc, yc, z_press))))
      kf.append((t + 0.75 * t_col, {}))
      kf.append((t + t_col, wrist_targets(w_face(xc, yc, EP_DEPTH + hover))))
      t += t_col
    if kf_only:
      return kf
    return keyframes(self, kf)
