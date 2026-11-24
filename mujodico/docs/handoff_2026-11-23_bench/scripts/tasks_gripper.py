"""OmniReset (arXiv 2603.15789) assembly tasks for the floating parallel gripper (embodiments.gripper_scene).

GripperPegInsert  square peg lying on the table -> grasp, rotate upright, insert into a fixed hole
GripperDrawer     drawer lying on its side -> grasp its ends, rotate it upright, set it down at the
                  cabinet, push it into the slot
GripperLegTwist   table leg lying on the table -> grasp, rotate upright, insert its stud into a socket
                  and twist so its key drives a nut (viscous hinge) by 90 deg

Each witness is open-loop controls refined offline by re-simulating under the true law (placement
offsets / tilt in the pads measured at a hover and corrected); witness(...) takes ~1-2 s.

Shared modelling choices (see each docstring for the full list):
  * Collision filter bits (as in bench_tasks): pads contype 1 (conaffinity 0); table conaffinity 2;
    fixture spheres contype 4; manipulated-object geoms list the bits they accept in conaffinity.
  * Concave fixtures (hole, socket, cabinet slot) are built from static spheres that collide with the
    manipulated object's box / capsule (sphere-box / sphere-capsule, far-field distances, no box-box).
    These fixture contacts are frictionless (condim 1 + priority 1: 1 contact dim instead of 4).
  * No joint dry friction (frictionloss): it made the true law blow up in jammed holds (see
    GripperLegTwist); the nut is purely viscous.
  * The pads never collide with the table: the wrist-height floor wz >= PAD_LOW keeps vertical pads
    above it (as in GripperCubeStack).  The pads never collide with the fixtures either (they stay
    above them in a sensible solution; it is a documented simplification, see each task).
  * Fixtures are seed-independent (simtools.sim_true caches one compiled model per task class);
    seeds only change the start state x_0.
"""
import numpy as np
import mujoco
from bench_tasks import register, BenchTask, pose_Q, _qpos_adr, _diag, yaw_quat, PAD_LOW
import embodiments as E
from embodiments import fmt, quat_mul, axis_quat

PAD_MID = 0.035      # pad-sphere midpoint below the wrist origin (pads at -0.045, -0.025)
TABLE_ENV = """    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" rgba="0.55 0.45 0.35 1"
          contype="0" conaffinity="2" friction="0.6"/>"""


def _mat2quat(R):
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
  return q


def _quat2mat(q):
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, np.asarray(q, float))
  return R.reshape(3, 3)


def _Rx(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _wrist_R(a, b, c):
  """Wrist orientation for hinge angles (wrx, wry, wrz) = Rx(a) Ry(b) Rz(c)."""
  return _Rx(a) @ _Ry(b) @ _Rz(c)


def _euler_xyz(R):
  """(a, b, c) with R = Rx(a) Ry(b) Rz(c) (the wrist hinge chain wrx, wry, wrz)."""
  b = np.arcsin(np.clip(R[0, 2], -1, 1))
  return np.arctan2(-R[1, 2], R[2, 2]), b, np.arctan2(-R[0, 1], R[0, 0])


def _axis_angle(axis, ang):
  return _quat2mat(axis_quat(axis, ang))


def _ring(prefix, center, zs, D, Rs, ct, n=4, phase=0.0, rgba='0.45 0.45 0.5 1'):
  """Static collision spheres at distance D (centres) from `center` along the n directions
  phase + 2 pi k / n, at heights zs: the (frictionless, condim 1) walls of a hole."""
  out = []
  for iz, z in enumerate(zs):
    for k in range(n):
      th = phase + 2 * np.pi * k / n
      p = (center[0] + D * np.cos(th), center[1] + D * np.sin(th), z)
      out.append(f'      <geom name="{prefix}{iz}{k}" type="sphere" size="{Rs}" pos="{fmt(p)}" mass="0" '
                 f'contype="{ct}" conaffinity="0" condim="1" priority="1" rgba="{rgba}"/>')
  return '\n'.join(out)


def _bar_body(name, a, hl, mass, rgba, friction=0.4, inertia=None):
  """Square bar (box a x a x hl half sizes, axis = body z).  The box collides with the pads (bit 1)
  and the fixture spheres (bit 4); two inscribed end spheres (radius a, at the end-face centres)
  collide with the table (bit 2).  The -z end carries a white cap (the end that is grasped)."""
  inert = '' if inertia is None else f'\n      <inertial pos="0 0 0" mass="{mass}" diaginertia="{fmt(inertia)}"/>'
  return f"""      <freejoint name="{name}"/>{inert}
      <geom name="{name}" type="box" size="{a} {a} {hl}" mass="{mass}" rgba="{rgba}" contype="0" conaffinity="5" friction="{friction}"/>
      <geom name="{name}_e1" type="sphere" size="{a}" pos="0 0 {hl - a}" mass="0" contype="2" conaffinity="0" friction="0.5" group="3" rgba="0 0 1 0.3"/>
      <geom name="{name}_e2" type="sphere" size="{a}" pos="0 0 {-(hl - a)}" mass="0" contype="2" conaffinity="0" friction="0.5" group="3" rgba="0 0 1 0.3"/>
      <geom name="{name}_cap" type="box" size="{a * 1.01} {a * 1.01} 0.004" pos="0 0 {-(hl - 0.004)}" mass="0" contype="0" conaffinity="0" rgba="1 1 1 1"/>"""


class _BarTask(BenchTask):
  """Common part of GripperPegInsert / GripperLegTwist: a square bar lying on the table (axis
  horizontal, roughly along world y), grasped near its -z end, rotated upright and inserted."""
  A = 0.015            # bar half width (3 cm square)
  HL = 0.06            # bar half length
  MASS = 0.06
  CLEAR = 0.003        # radial clearance per side between bar face and hole wall
  RS = 0.015           # hole-wall sphere radius
  RING_Z = (0.015, 0.045)
  HOLE = np.array([-0.10, 0.10])
  GRASP_FROM_END = 0.025   # pad centre distance from the grasped (-z) end
  T_INSERT = 0.6           # witness: duration of the final descent into the hole
  T_TWIST = 1.3            # witness: duration of the twist
  TWIST_LEAD = 0.1         # witness: twist targets lead the path by this much (wrist servo lag)
  HORIZON = 100
  DT = 0.05

  def _start(self, seed):
    rng = np.random.default_rng(self.SEED_BASE + seed)
    p = np.array([0.12 + rng.uniform(-0.03, 0.03), -0.08 + rng.uniform(-0.03, 0.03), self.A - 1e-4])
    dyaw = rng.uniform(-0.4, 0.4)
    q = _mat2quat(_Rz(dyaw) @ _Rx(np.pi / 2))     # body z (bar axis) -> (sin d, -cos d, 0)
    return p, q, dyaw

  def _ranges(self):
    return ((-0.3, 0.3), (-0.3, 0.3), (PAD_LOW, 0.35), (-0.5, 2.0), (-1.6, 1.6), (-3.2, 3.2))

  # --- witness geometry -------------------------------------------------------------------------
  def _grasp_kf(self):
    """Keyframes: approach, grasp the bar near its -z end, lift.  Returns (kf, t_end, s_g)."""
    m = self.model
    a = _qpos_adr(m, self.BAR)
    p0 = self.x_0[a:a + 3]
    R0 = _quat2mat(self.x_0[a + 3:a + 7])
    zdir = R0[:, 2]
    s_g = self.HL - self.GRASP_FROM_END                  # grasp point at -s_g along the bar axis
    g = p0 - s_g * zdir
    c0 = np.arctan2(zdir[1], zdir[0])                    # gripper x along the bar axis (+z end)
    wz = max(PAD_LOW, self.A + PAD_MID)
    kf = [
        (0.6, dict(a_wx=g[0], a_wy=g[1], a_wz=wz + 0.06, a_wrz=c0)),
        (0.95, dict(a_wz=wz)),
        (1.2, dict(a_grip=0.0)),
        (1.3, {}),
        (1.65, dict(a_wz=0.20)),
    ]
    return kf, 1.65, s_g, c0

  def _upright_wrist(self, grasp_pt, b, Rc=None, phi=0.0, pivot=None):
    """Wrist targets holding the bar upright (+z end down) with the pad centre at grasp_pt:
    nominal orientation Rx(pi/2) Ry(b) Rz(-pi/2) (approach horizontal, wrist yawed by b about world
    z), pre-multiplied by a tilt correction Rc (about the pad centre), then the whole pose turned by
    phi about the vertical axis through `pivot` (the twist)."""
    Rc = np.eye(3) if Rc is None else Rc
    Rw = Rc @ _wrist_R(np.pi / 2, b, -np.pi / 2)
    w = np.asarray(grasp_pt, float) - PAD_MID * (Rw @ np.array([0, 0, -1.0]))
    if phi:
      Rz = _Rz(phi)
      Rw = Rz @ Rw
      pv = np.array([*pivot, 0.0])
      w = pv + Rz @ (w - pv)
    a, bb, c = _euler_xyz(Rw)
    return dict(a_wx=w[0], a_wy=w[1], a_wz=w[2], a_wrx=a, a_wry=bb, a_wrz=c)

  def _bar_witness(self, b0, twist=0.0, play=0.0, back=0.0, iters=3, release=False):
    """Grasp, lift, rotate upright over the hole (wrist yaw b0 about world z), insert, optionally twist
    by `twist` (+ `play` overshoot to take up the key's play, then `back`).  The open-loop controls are
    refined by re-simulating under the true law: at the pre-insertion hover the measured tilt of the
    bar in the pads is corrected by a wrist rotation and the offset of its lower end from the hole
    axis by a wrist translation."""
    from simtools import keyframes, sim_true
    m = self.model
    ba = _qpos_adr(m, self.BAR)
    hole = np.asarray(self.HOLE, float)
    dxy = np.zeros(2); Rc = np.eye(3)
    for it in range(iters + 1):
      kf, t, s_g, c0 = self._grasp_kf()
      def W(z, **kw):
        g = np.array([hole[0] + dxy[0], hole[1] + dxy[1], z])
        return self._upright_wrist(g, b0, Rc, **kw)
      z_down = self.HL + s_g - 0.004
      wh = W(self.HL + s_g + 0.085)       # lower end 2.5 cm above the hole rim
      wp = W(self.HL + s_g + 0.072)       # lower end 1.2 cm above the hole rim
      wd = W(z_down)
      t_h = t + 1.15
      kf += [(t + 0.8, wh), (t + 1.0, wp), (t_h, {}), (t + 1.15 + self.T_INSERT, wd), (t + 1.25 + self.T_INSERT, {})]
      t += 1.25 + self.T_INSERT
      if twist:
        b1 = twist + play
        n = 20
        lead = self.TWIST_LEAD             # servo lag compensation: targets lead the path by kv/kp
        for i in range(1, n + 1):          # twist (smoothstep): the wrist circles the hole (= bar) axis
          sgm = min(1.0, i / n + lead / self.T_TWIST)
          kf.append((t + self.T_TWIST * i / n, W(z_down, phi=b1 * sgm * sgm * (3 - 2 * sgm), pivot=hole)))
        t += self.T_TWIST
        if back:
          kf += [(t + 0.35, W(z_down, phi=b1 - back, pivot=hole)), (t + 0.5, {})]
          t += 0.5
      if release:                          # open the jaws and back the wrist away
        kf += [(t + 0.25, dict(a_grip=0.03))]
        wr = W(z_down + 0.03, phi=twist + play, pivot=hole)
        app = np.array([wr['a_wx'], wr['a_wy']]) - hole
        wr['a_wx'], wr['a_wy'] = hole + app * (1 + 0.03 / np.linalg.norm(app))
        kf += [(t + 0.6, wr)]
      us = keyframes(self, kf)
      if it == iters:
        return us
      X = sim_true(self, us)
      k = int(round(t_h / self.DT))
      Rb = _quat2mat(X[k, ba + 3:ba + 7])
      zb = Rb[:, 2]                                   # points down along the bar
      ax = np.cross(zb, [0, 0, -1.0]); sn = np.linalg.norm(ax)
      if sn > 1e-9:
        Rc = _axis_angle(ax / sn, np.arctan2(sn, -zb[2])) @ Rc
      bot = X[k, ba:ba + 3] + zb * self.HL           # +z end (down)
      dxy = dxy - (bot[:2] - hole)

  def _bar_goal_quat(self, spin):
    return _mat2quat(_Rz(spin) @ _Rx(np.pi))


@register
class GripperPegInsert(_BarTask):
  """OmniReset 'Peg Insertion': grasp a square peg lying on the table, reorient it upright (a 90 deg
  wrist rotation about the jaw axis) and insert it into a fixed hole to the floor (4.5 cm deep).

  Model.  Peg: 3 x 3 x 12 cm square bar, 60 g (box collides with pads and hole walls; two inscribed
  end spheres r = 1.5 cm collide with the table).  The peg starts lying on a face, axis along world
  y +- 0.4 rad, at (0.12, -0.08) +- 3 cm (seeded).  Hole (fixed, centre (-0.10, 0.10)): two rings
  of four static spheres (r = 1.5 cm, centres at z = 1.5 and 4.5 cm, facing the four peg faces) --
  a square-ish hole 4.5 cm deep with rounded (chamfered) walls, radial clearance 3 mm per side at the
  face centres (~4.5 mm along the diagonals; the peg can spin +-23 deg about its axis inside the
  hole before a corner jams).  The hole floor is the table.  Adjacent ring spheres leave a 1.7 cm gap
  (< 3 cm peg width) and the rings touch vertically, so the peg can only enter from above.
  Contacts (14; 32 contact dims): 4 pad spheres vs peg box; 2 peg end spheres vs table; 8 hole
  spheres vs peg box (frictionless, condim 1).
  Simplifications: pads vs table not modelled (wrist floor wz >= PAD_LOW); pads vs hole fixture not
  modelled (in the upright hold the pads are 3 cm above the fixture); the peg's box edges vs table
  are replaced by the inscribed end spheres (a lying peg rolls about its axis like a cylinder and its
  corners may dip ~4 mm into the table when strongly tilted); the fixture block between the rings has
  no collision (only the 8 spheres exist).
  Goal: peg upright with its capped (-z) end UP, faces aligned with the hole (one of the 8
  symmetric orientations of a square peg, the one reached by grasping the capped end and rotating
  the free end down), bottom on the hole floor.  Cost: peg pose (pos 100, quat 1); success < 0.01
  (~1 cm / ~11 deg)."""
  BAR = 'peg'
  SEED_BASE = 6100
  HORIZON = 90

  def __init__(self, seed=0):
    a, hl = self.A, self.HL
    p0, q0, self.dyaw = self._start(seed)
    D = a + self.CLEAR + self.RS
    env = f"""{TABLE_ENV}
{_ring('hole', self.HOLE, self.RING_Z, D, self.RS, ct=4)}
    <geom name="hole_plate" type="box" size="0.055 0.055 0.0005" pos="{fmt([*self.HOLE, 0.0005])}" contype="0" conaffinity="0" rgba="0.3 0.3 0.35 1"/>
    <body name="peg" pos="{fmt(p0)}" quat="{fmt(q0)}">
{_bar_body('peg', a, hl, self.MASS, '0.85 0.55 0.2 1')}
    </body>"""
    m = E.gripper_model(env, wrist_pos=np.zeros(3), ranges=self._ranges(), timestep=self.DT)
    x0 = np.zeros(m.nq + m.nv)
    pa = _qpos_adr(m, 'peg')
    x0[pa:pa + 3] = p0; x0[pa + 3:pa + 7] = q0
    w = _qpos_adr(m, 'wx')
    x0[w:w + 6] = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
    x0[_qpos_adr(m, 'jl')] = x0[_qpos_adr(m, 'jr')] = 0.04
    g = x0.copy()
    g[pa:pa + 3] = [*self.HOLE, hl - 1e-4]
    g[pa + 3:pa + 7] = self._bar_goal_quat(0.0)
    Q = pose_Q(m, 'peg')
    u_ref = np.concatenate([x0[w:w + 6], [0.04] * (m.nu - 6)])   # one target per jaw actuator
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-1] * (m.nu - 6))
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)

  def witness(self):
    return self._bar_witness(b0=0.0)


@register
class GripperLegTwist(_BarTask):
  """OmniReset 'Screwing in a table leg': pick up a table leg lying on the table, stand it upright
  in a socket and twist it so that its key turns the socket's nut ("thread") by 90 deg.

  Model.  Leg (60 g, 14 cm): upper half a 3 x 3 cm square bar (the grip section; collides with the
  pads only), lower half a round stud (capsule r = 1.5 cm; collides with the socket rings only); two
  end spheres r = 1.5 cm (the stud's end cap / inscribed in the square end) collide with the table.
  A key pin (sphere r = 6 mm, centre 2.1 cm off the axis, 1.5 cm above the lower (+z) end; it sticks
  out 1.2 cm) collides with the nut's dogs only.  The leg starts lying on a face, axis along world
  y +- 0.4 rad, at (0.12, -0.08) +- 3 cm (seeded).
  Socket at (-0.10, 0.10): a fixed round hole made of two rings of four static spheres (r = 1.2 cm,
  centres at z = 3.5 and 5.8 cm, radial clearance 3 mm to the stud) above a NUT: a body on a hinge
  about world z with viscous damping 0.02 N m s and armature 2e-3, carrying two dog pins (vertical
  capsules r = 5 mm at radius 2.2 cm, z 0.4..2.6 cm, +-35 deg about the slot centre) that form a
  slot for the key (+-0.15 rad of play).  The slot starts at -45 deg (the leg is then inserted with
  the wrist yawed -45 deg and the 90 deg twist stays inside the wrist's +-1.6 rad range).  Turning
  the seated leg drives the nut through key-vs-dog contacts only: the gripper cannot touch the nut.
  Contacts (16; 34 contact dims): 4 pad spheres vs leg grip section; 2 leg end spheres vs table;
  8 ring spheres vs stud and key vs 2 dogs (these 10 frictionless, condim 1).
  Simplifications: the thread is a pure rotation with viscous resistance -- no axial advance and NO
  dry friction: a frictionloss on the nut hinge (0.01-0.03 N m) made the true law blow up (single-step
  explosions of the held leg, at rest, right after the twist) whenever the key was loaded against a
  dog while the gripper held the leg in the rings, so the nut holds its angle only by damping;
  the tabletop is the work table (socket floor); the key does not collide with the ring spheres (the
  ring gaps 45 deg from its spheres act as the keyway: the witness inserts the key through a gap,
  where it clears the spheres physically, but a solver could push it through a ring sphere); the
  pads do not collide with the round stud and the grip section does not collide with the rings
  (inserting the square end first would pass through the rings, but cannot reach the goal pose);
  pads vs table not modelled (wrist floor wz >= PAD_LOW); pads vs socket not modelled (the pads hold
  the leg ~4 cm above the rings).
  Goal: nut angle +90 deg (joint 'nut' = pi/2) and the leg seated upright, stud end down, key
  centred in the turned slot.  Cost: leg pose (pos 100, quat 0.5) + nut angle 1; success < 0.01
  (~1 cm, ~6 deg of nut angle)."""
  BAR = 'leg'
  SEED_BASE = 6300
  HL = 0.07
  MASS = 0.06
  NUT0 = -np.pi / 4
  TWIST = np.pi / 2
  RING_Z = (0.035, 0.058)
  RS = 0.012
  KEY_RHO, KEY_R, KEY_Z = 0.021, 0.006, 0.015     # key pin: radius from axis, size, height above the end
  DOG_RHO, DOG_R, DOG_ANG = 0.022, 0.005, 0.61     # dog pins (vertical capsules z 0.004..0.026)
  NUT_FL = 0.0
  NUT_DAMP = 0.02
  NUT_ARM = 0.002
  HORIZON = 120

  def __init__(self, seed=0):
    r, hl = self.A, self.HL
    p0, q0, self.dyaw = self._start(seed)
    D = r + self.CLEAR + self.RS
    dogs = '\n'.join(
        f'      <geom name="dog{i}" type="capsule" size="{self.DOG_R}" fromto="{fmt([self.DOG_RHO * np.cos(a), self.DOG_RHO * np.sin(a), 0.004, self.DOG_RHO * np.cos(a), self.DOG_RHO * np.sin(a), 0.026])}" '
        f'mass="0" contype="0" conaffinity="8" condim="1" priority="1" rgba="0.9 0.2 0.1 1"/>'
        for i, a in enumerate((self.DOG_ANG, -self.DOG_ANG)))
    env = f"""{TABLE_ENV}
{_ring('ring', self.HOLE, self.RING_Z, D, self.RS, ct=4, phase=self.NUT0 + np.pi / 4)}
    <body name="nut" pos="{fmt([*self.HOLE, 0.0])}" quat="{fmt(yaw_quat(self.NUT0))}">
      <joint name="nut" type="hinge" axis="0 0 1" frictionloss="{self.NUT_FL}" damping="{self.NUT_DAMP}" armature="{self.NUT_ARM}"/>
      <geom name="nut_disc" type="cylinder" size="0.04 0.002" pos="0 0 0.002" mass="0.1" contype="0" conaffinity="0" rgba="0.6 0.6 0.65 1"/>
      <geom name="nut_mark" type="box" size="0.035 0.003 0.001" pos="0 0 0.005" mass="0" contype="0" conaffinity="0" rgba="0.9 0.1 0.1 1"/>
{dogs}
    </body>
    <body name="leg" pos="{fmt(p0)}" quat="{fmt(q0)}">
      <freejoint name="leg"/>
      <geom name="leg" type="box" size="{r} {r} {hl / 2}" pos="0 0 {-hl / 2}" mass="{self.MASS * 0.6}" rgba="0.6 0.4 0.25 1" contype="0" conaffinity="1" friction="0.4"/>
      <geom name="leg_stud" type="capsule" size="{r} {hl / 2 - r}" pos="0 0 {hl / 2}" mass="{self.MASS * 0.4}" rgba="0.7 0.7 0.72 1" contype="0" conaffinity="4" friction="0.4"/>
      <geom name="leg_e1" type="sphere" size="{r}" pos="0 0 {hl - r}" mass="0" contype="2" conaffinity="0" friction="0.5" group="3" rgba="0 0 1 0.3"/>
      <geom name="leg_e2" type="sphere" size="{r}" pos="0 0 {-(hl - r)}" mass="0" contype="2" conaffinity="0" friction="0.5" group="3" rgba="0 0 1 0.3"/>
      <geom name="leg_key" type="sphere" size="{self.KEY_R}" pos="{self.KEY_RHO} 0 {hl - self.KEY_Z}" mass="0" contype="8" conaffinity="0" friction="0.3" rgba="0.9 0.9 0.1 1"/>
      <geom name="leg_cap" type="cylinder" size="{r * 1.02} 0.004" pos="0 0 {-(hl - 0.012)}" mass="0" contype="0" conaffinity="0" rgba="1 1 1 1"/>
    </body>"""
    m = E.gripper_model(env, wrist_pos=np.zeros(3), ranges=self._ranges(), timestep=self.DT)
    nx = m.nq + m.nv
    x0 = np.zeros(nx)
    la = _qpos_adr(m, 'leg'); na = _qpos_adr(m, 'nut')
    x0[la:la + 3] = p0; x0[la + 3:la + 7] = q0
    w = _qpos_adr(m, 'wx')
    x0[w:w + 6] = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
    x0[_qpos_adr(m, 'jl')] = x0[_qpos_adr(m, 'jr')] = 0.04
    g = x0.copy()
    g[la:la + 3] = [*self.HOLE, hl - 1e-4]
    g[la + 3:la + 7] = self._bar_goal_quat(self.NUT0 + self.TWIST)
    g[na] = self.TWIST
    Q = pose_Q(m, 'leg', w_rot=0.5) + _diag(nx, {na: 1.0})
    u_ref = np.concatenate([x0[w:w + 6], [0.04] * (m.nu - 6)])   # one target per jaw actuator
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-1] * (m.nu - 6))
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)

  def witness(self, play=0.21, back=0.15, iters=3, release=False):
    return self._bar_witness(b0=self.NUT0, twist=self.TWIST, play=play, back=back, iters=iters, release=release)


DRAWER_H = (0.035, 0.03, 0.02)     # drawer half sizes (7 x 6 x 4 cm; handle on the +x face)


def _drawer_corners(h, r):
  """Corner spheres of the faces the drawer rests on: the -y side face (start) and the -z bottom."""
  out = []
  for sx in (-1, 1):
    for sy, sz in ((-1, -1), (-1, 1), (1, -1)):
      p = (sx * (h[0] - r), sy * (h[1] - r), sz * (h[2] - r))
      out.append(f'      <geom name="drw_c{sx:+d}{sy:+d}{sz:+d}" type="sphere" size="{r}" pos="{fmt(p)}" mass="0" '
                 f'contype="2" conaffinity="0" friction="0.5" group="3" rgba="0.2 0.2 0.8 0.5"/>')
  return '\n'.join(out)


@register
class GripperDrawer(BenchTask):
  """OmniReset 'Drawer insertion': a drawer lying on its side must be flipped upright and pushed
  into a cabinet slot (handle out).

  Model.  Drawer: solid 7 x 6 x 4 cm box, 100 g, handle (visual) on its +x face.  It starts lying
  on its -y side face (6 cm tall, 4 cm thick), yaw +- 0.3 rad, at (0.10, -0.12) +- 3 cm (seeded).
  Collision: the box collides with the pads and with the cabinet; six corner spheres (r = 4 mm) of
  its -y side face and -z bottom face collide with the table (the corners it can rest or pivot on
  while flipping from the side onto the bottom).  Cabinet (fixed, slot opening facing +x, slot
  7 cm deep, centred at y = 0.12, mouth at x = -0.06): six static spheres (r = 1.2 cm) -- two at the
  mouth and two at the back of each side wall (4 mm clearance to the drawer's sides), one back stop
  and one ceiling sphere at the mouth (4 mm above the drawer).  The slot floor is the table.  The wall
  spheres leave <= 2.2 cm gaps (< the drawer's 4 cm thickness), so the drawer can only enter through
  the mouth, and only upright (the ceiling sphere stops a drawer on its side).
  Contacts (16; 46 contact dims): 4 pad spheres vs drawer box; 6 drawer corner spheres vs table;
  6 cabinet spheres vs drawer box (frictionless, condim 1).
  Simplifications: pads vs table not modelled (wrist floor wz >= PAD_LOW); pads vs cabinet not
  modelled (a solver could carry the drawer fully in with the pads passing through the side walls;
  the witness sets it down in front of the mouth and pushes it in); the drawer's top-face corners and the +y
  side face's top corners have no table contact (a drawer flipped onto its top or other side would
  sink 4 cm into the table -- unphysical but useless for reaching the goal); the cabinet body
  between its spheres has no collision.
  Goal: drawer upright (bottom down, handle facing +x) fully inserted: back face on the back stop,
  front face flush with the cabinet mouth.  Cost: drawer pose (pos 100, quat 1); success < 0.01
  (~1 cm / ~11 deg).
  Witness: grasp the lying drawer by its end faces (jaws along its 7 cm axis, 8.5 cm stroke; the jaws
  close only after the wrist has settled at the grasp height -- closing while it still descended
  froze the grasp 2 cm high, the upper pad spheres then sat on the face edge and the drawer pivoted
  in the pads), rotate it upright about the jaw axis (wrx -> -90 deg), set it down 1.2 cm in front of
  the mouth (horizontal pads at wrist z = W_SET, released from ~3 cm), back off, then push its front
  face with both vertical pads (jaws 2 cm open, wrist z = W_PUSH: pad spheres at 1.9 / 3.9 cm on the
  4 cm face).  The set-down point and the wrist yaw are refined under the true law from the drawer's
  pose after it has settled on the table.  The wrist stays >= 9 mm above its floor (wz >= PAD_LOW)
  throughout: the witness does not use the joint limit as a support (2026-11-24; it used to push
  with the wrist on the floor).  (Tipping it over by pushing its top does not work under the true
  law: the regularized friction lets it slide instead of pivot.)"""
  HORIZON = 120
  DT = 0.05
  SLOT_Y = 0.12
  MOUTH_X = -0.06
  CLEAR = 0.004
  RS = 0.012
  SEED_BASE = 6200

  def __init__(self, seed=0):
    h = DRAWER_H
    rng = np.random.default_rng(self.SEED_BASE + seed)
    yaw = rng.uniform(-0.3, 0.3)
    p0 = np.array([0.10 + rng.uniform(-0.03, 0.03), -0.12 + rng.uniform(-0.03, 0.03), h[1] - 1e-4])
    q0 = _mat2quat(_Rz(yaw) @ _Rx(np.pi / 2))         # body y up, body -z towards world +y
    xb = self.MOUTH_X - 2 * h[0]                        # slot back
    wy = h[1] + self.CLEAR + self.RS
    sph = [('mouthL', (self.MOUTH_X - self.RS, self.SLOT_Y + wy, h[2])),
           ('mouthR', (self.MOUTH_X - self.RS, self.SLOT_Y - wy, h[2])),
           ('backL', (xb + self.RS, self.SLOT_Y + wy, h[2])),
           ('backR', (xb + self.RS, self.SLOT_Y - wy, h[2])),
           ('stop', (xb - self.RS, self.SLOT_Y, h[2])),
           ('ceil', (self.MOUTH_X - self.RS, self.SLOT_Y, 2 * h[2] + self.CLEAR + self.RS))]
    cab = '\n'.join(f'    <geom name="cab_{n}" type="sphere" size="{self.RS}" pos="{fmt(p)}" contype="4" conaffinity="0" '
                    f'condim="1" priority="1" rgba="0.45 0.35 0.25 1"/>' for n, p in sph)
    wt = 0.008      # visual cabinet walls
    cabv = (f'    <geom name="cabv_l" type="box" size="{h[0]} {wt / 2} 0.03" pos="{fmt([xb + h[0], self.SLOT_Y + wy + self.RS - wt / 2, 0.03])}" contype="0" conaffinity="0" rgba="0.6 0.5 0.4 0.25"/>\n'
            f'    <geom name="cabv_r" type="box" size="{h[0]} {wt / 2} 0.03" pos="{fmt([xb + h[0], self.SLOT_Y - wy - self.RS + wt / 2, 0.03])}" contype="0" conaffinity="0" rgba="0.6 0.5 0.4 0.25"/>\n'
            f'    <geom name="cabv_t" type="box" size="{h[0]} {wy + self.RS} {wt / 2}" pos="{fmt([xb + h[0], self.SLOT_Y, 2 * h[2] + self.CLEAR + 2 * self.RS])}" contype="0" conaffinity="0" rgba="0.6 0.5 0.4 0.25"/>')
    env = f"""{TABLE_ENV}
{cab}
{cabv}
    <body name="drawer" pos="{fmt(p0)}" quat="{fmt(q0)}">
      <freejoint name="drawer"/>
      <geom name="drawer" type="box" size="{fmt(h)}" mass="0.1" rgba="0.8 0.65 0.35 1" contype="0" conaffinity="5" friction="0.5"/>
{_drawer_corners(h, 0.004)}
      <geom name="drawer_handle" type="box" size="0.003 0.015 0.004" pos="{h[0] + 0.003} 0 0.005" mass="0" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
      <geom name="drawer_top" type="box" size="{h[0] * 0.9} {h[1] * 0.9} 0.0005" pos="0 0 {h[2]}" mass="0" contype="0" conaffinity="0" rgba="1 1 1 1"/>
    </body>"""
    ranges = ((-0.3, 0.3), (-0.3, 0.3), (PAD_LOW, 0.35), (-1.6, 1.6), (-1.6, 1.6), (-3.2, 3.2))
    m = E.gripper_model(env, wrist_pos=np.zeros(3), ranges=ranges, timestep=self.DT)
    x0 = np.zeros(m.nq + m.nv)
    da = _qpos_adr(m, 'drawer')
    x0[da:da + 3] = p0; x0[da + 3:da + 7] = q0
    w = _qpos_adr(m, 'wx')
    x0[w:w + 6] = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
    x0[_qpos_adr(m, 'jl')] = x0[_qpos_adr(m, 'jr')] = 0.04
    g = x0.copy()
    g[da:da + 3] = [xb + h[0], self.SLOT_Y, h[2] - 1e-4]
    g[da + 3:da + 7] = [1, 0, 0, 0]
    Q = pose_Q(m, 'drawer')
    u_ref = np.concatenate([x0[w:w + 6], [0.04] * (m.nu - 6)])   # one target per jaw actuator
    R = np.diag([1e-2] * 3 + [1e-3] * 3 + [1e-1] * (m.nu - 6))
    super().__init__(model=m, horizon=self.HORIZON, x_0=x0, x_goal=g, Q=Q, Qf=40 * Q,
                     u_ref=u_ref, R=R, contact_smoothing=0.002)

  W_SET = PAD_LOW + 0.013     # witness: wrist z (= horizontal pad centre) when setting the drawer down
  W_PUSH = PAD_LOW + 0.013    # witness: wrist z while pushing with the vertical pads
  T_MEAS = 3.6                # witness: time at which the settled drawer's pose is measured

  def _drawer_kf(self, place, dyaw=0.0):
    """Grasp the lying drawer by its end faces (jaws along its x axis, 1.5 cm above its centre in
    the upright frame), rotate it upright about the jaw axis, set it down at `place` (xy) in front of
    the slot, release, and push it in with the two vertical pads on its front face."""
    m = self.model; h = DRAWER_H
    da = _qpos_adr(m, 'drawer')
    p0 = self.x_0[da:da + 3]; R0 = _quat2mat(self.x_0[da + 3:da + 7])
    off = 0.015                                         # grasp offset along body z (up when upright)
    g = p0 + off * R0[:, 2]
    c0 = np.arctan2(R0[1, 0], R0[0, 0]) - np.pi / 2     # gripper y along body x
    def upright(pc, z):
      return dict(a_wx=pc[0], a_wy=pc[1] + PAD_MID, a_wz=z, a_wrx=-np.pi / 2, a_wry=-dyaw, a_wrz=-np.pi / 2)
    face = self.x_goal[da] + h[0]                        # goal x of the front face
    kf = [
        (0.6, dict(a_wx=g[0], a_wy=g[1], a_wz=0.15, a_wrz=c0, a_grip=0.0425)),
        (0.95, dict(a_wz=g[2] + PAD_MID)),
        (1.15, {}),                                     # let the wrist settle before closing
        (1.35, dict(a_grip=0.0)),
        (1.45, {}),
        (1.75, dict(a_wz=0.16)),
        (2.55, upright(place, 0.12)),
        (2.95, upright(place, self.W_SET)),
        (3.05, {}),
        (3.25, dict(a_grip=0.0425)),                    # release
        (3.45, dict(a_wy=place[1] + PAD_MID + 0.03)),   # pull the pads out along +y
        (3.65, dict(a_wz=0.10)),
        (4.05, dict(a_wx=place[0] + h[0] + 0.035, a_wy=self.SLOT_Y, a_wz=0.10, a_wrx=0.0, a_wry=0.0, a_wrz=0.0, a_grip=0.02)),
        (4.3, dict(a_wz=self.W_PUSH)),
        (5.2, dict(a_wx=face + 0.006 - 0.004)),         # push the front face to the mouth
        (5.4, {}),
    ]
    return kf

  def witness(self, iters=3):
    from simtools import keyframes, sim_true
    m = self.model; h = DRAWER_H
    da = _qpos_adr(m, 'drawer')
    place0 = np.array([self.MOUTH_X + h[0] + 0.012, self.SLOT_Y])
    place = place0.copy(); dyaw = 0.0
    for it in range(iters + 1):
      us = keyframes(self, self._drawer_kf(place, dyaw))
      if it == iters:
        return us
      X = sim_true(self, us)
      k = int(round(self.T_MEAS / self.DT))           # drawer released and settled on the table
      err = X[k, da:da + 2] - place0
      Rk = _quat2mat(X[k, da + 3:da + 7])
      place = place - err
      dyaw = dyaw + np.arctan2(Rk[1, 0], Rk[0, 0])
