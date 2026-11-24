"""Shared embodiments for the 2026-11-23 hard-task benchmark.

Two floating, gravity-compensated manipulators stand in for the papers' arm + end-effector:
  * `allegro_scene`: Allegro hand (the repo's right_hand2.xml, thumb tip collision enabled) on a
    6-DoF position-servoed wrist (x, y, z slides, then hinges about x, y, z).  SimToolReal uses a
    22-DoF Sharpa hand on a 7-DoF iiwa (29 DoF); here 6 + 16 = 22 actuated DoF.
  * `gripper_scene`: parallel-jaw gripper (Robotiq 2F-85-like, 85 mm stroke) on the same 6-DoF
    wrist; two independently servoed jaws (a_grip, a_grip_r; coupled=True: joint equality + one actuator).
    OmniReset uses a UR7e + Robotiq 2F-85 (6 + 1 DoF); here 6 + 1 actuated DoF.

Wrist targets are absolute poses (position servo): holding still is u = current wrist pose.
Collision geometry is primitives only (spheres / capsules / boxes); see each task for its
contact filter (contype / conaffinity bits):
  bit 1 (1): hand / gripper links  (conaffinity 1, contype 0 on the Allegro links)
  bit 2 (2): environment (table, fixtures)
  bit 3 (4): manipulated objects among themselves
"""
import re, tempfile
from pathlib import Path
import numpy as np
import mujoco
from mujodico.examples.tasks import allegro_ball as _ab
import table_contacts   # orientation-complete table contacts (BENCH_TABLE=legacy: off)

ALLEGRO_DIR = Path(_ab.__file__).resolve().parent / 'models' / 'wonik_allegro'
WRIST_JOINTS = ('wx', 'wy', 'wz', 'wrx', 'wry', 'wrz')


def quat_mul(a, b):
  w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
  return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                   w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def axis_quat(axis, ang):
  axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
  return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def fmt(v):
  return ' '.join(f'{float(x):.6g}' for x in v)


def from_xml_in(dirpath, xml):
  """Compile an MJCF string as if it lived in `dirpath` (so includes / meshdir resolve)."""
  with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', dir=dirpath, delete=False) as f:
    f.write(xml)
  try:
    if not table_contacts.enabled():
      return mujoco.MjModel.from_xml_path(f.name)
    return table_contacts.apply(mujoco.MjSpec.from_file(f.name)).compile()
  finally:
    Path(f.name).unlink()


def _wrist_block(pos, ranges, slide_arm=0.2, hinge_arm=0.01):
  (xr, yr, zr, rxr, ryr, rzr) = ranges
  return f"""
    <body name="wrist" pos="{fmt(pos)}" gravcomp="1">
      <joint name="wx" type="slide" axis="1 0 0" range="{fmt(xr)}" armature="{slide_arm}" damping="0"/>
      <joint name="wy" type="slide" axis="0 1 0" range="{fmt(yr)}" armature="{slide_arm}" damping="0"/>
      <joint name="wz" type="slide" axis="0 0 1" range="{fmt(zr)}" armature="{slide_arm}" damping="0"/>
      <joint name="wrx" type="hinge" axis="1 0 0" range="{fmt(rxr)}" armature="{hinge_arm}" damping="0"/>
      <joint name="wry" type="hinge" axis="0 1 0" range="{fmt(ryr)}" armature="{hinge_arm}" damping="0"/>
      <joint name="wrz" type="hinge" axis="0 0 1" range="{fmt(rzr)}" armature="{hinge_arm}" damping="0"/>
      <geom name="wrist_mass" type="sphere" size="0.01" mass="0.05" contype="0" conaffinity="0" rgba="0.3 0.3 0.3 0.3"/>
"""


def _wrist_actuators(ranges, kp_lin, kv_lin, f_lin, kp_ang, kv_ang, f_ang):
  out = []
  for j, r in zip(WRIST_JOINTS, ranges):
    lin = j in ('wx', 'wy', 'wz')
    kp, kv, f = (kp_lin, kv_lin, f_lin) if lin else (kp_ang, kv_ang, f_ang)
    out.append(f'    <position name="a_{j}" joint="{j}" kp="{kp}" kv="{kv}" ctrlrange="{fmt(r)}" '
               f'forcelimited="true" forcerange="{-f} {f}"/>')
  return '\n'.join(out)


DEFAULT_RANGES = ((-0.4, 0.4), (-0.4, 0.4), (0.0, 0.5), (-3.2, 3.2), (-1.6, 1.6), (-3.2, 3.2))


def allegro_scene(env_world, *, wrist_pos, palm_quat, ranges=DEFAULT_RANGES, env_assets='', env_default='',
                  extra_contact='', extra_actuators='', extra_equality='', timestep=0.04, thumb=True,
                  finger_kp=1.0, finger_force=2.0, links=False, hand_friction=None, option_extra=''):
  """Allegro hand on a 6-DoF wrist.  `palm_quat` = palm orientation in the wrist frame at zero
  wrist angles (the repo's palm frame: fingers along local +z, palm normal local +x).
  links=True also enables the proximal/medial link capsules (otherwise only tips + palm collide)."""
  src = (ALLEGRO_DIR / 'right_hand2.xml').read_text()
  default = re.search(r'<default>.*</default>', src, re.S).group(0)
  asset = re.search(r'<asset>.*?</asset>', src, re.S).group(0)
  palm = re.search(r'(<body name="palm".*?)\n  </worldbody>', src, re.S).group(1)
  contact = re.search(r'<contact>(.*?)</contact>', src, re.S).group(1)
  acts = re.search(r'<actuator>(.*?)</actuator>', src, re.S).group(1)
  if thumb:   # thumb tip collides like the other tips
    default = default.replace('<geom group="4" contype="0" conaffinity="0" type="capsule" size="0.012 0.010" pos="0 0 0.033"/>',
                              '<geom type="capsule" size="0.012 0.010" pos="0 0 0.033"/>')
  if links:
    for cls in ('proximal_collision', 'medial_collision'):
      default = re.sub(rf'(<default class="{cls}">\s*<geom) group="4" contype="0" conaffinity="0"', r'\1', default)
  if hand_friction is not None:
    default = default.replace('<geom group="3" contype="0" conaffinity="1" mass="0" rgba="1 0 0 1"/>',
                              f'<geom group="3" contype="0" conaffinity="1" mass="0" rgba="1 0 0 1" friction="{hand_friction}"/>')
  default = default.replace('<position kp="1" forcelimited="true" forcerange="-2 2"/>',
                            f'<position kp="{finger_kp}" forcelimited="true" forcerange="{-finger_force} {finger_force}"/>')
  palm = palm.replace('<body name="palm" quat="0 1 0 1"', f'<body name="palm" quat="{fmt(palm_quat)}"')
  palm = re.sub(r'<body name="([a-z_]+)"', r'<body name="\1" gravcomp="1"', palm)
  palm = '\n'.join('  ' + l for l in palm.splitlines())
  xml = f"""<mujoco model="allegro bench">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" iterations="40" tolerance="1e-6" {option_extra}/>
  {default}
  {env_default}
  {asset}
  <asset>{env_assets}</asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
{env_world}
{_wrist_block(wrist_pos, ranges)}
{palm}
    </body>
  </worldbody>
  <contact>{contact}{extra_contact}</contact>
  <equality>{extra_equality}</equality>
  <actuator>
{_wrist_actuators(ranges, 400, 40, 40, 8, 0.6, 4)}
{acts}
{extra_actuators}
  </actuator>
</mujoco>
"""
  return xml


def allegro_model(env_world, **kw):
  return from_xml_in(ALLEGRO_DIR, allegro_scene(env_world, **kw))


# ---------------------------------------------------------------------------------------------
# Parallel-jaw gripper
# ---------------------------------------------------------------------------------------------
# Gripper frame (wrist frame at zero angles): approach direction -z (fingers point down),
# jaws close along y.  Each pad is a box (visual + collision against boxes) with three
# collision spheres along its length (sphere-box / sphere-sphere give far-field distances).
GRIP_STROKE = 0.0425   # per-jaw travel (85 mm total opening)


def gripper_scene(env_world, *, wrist_pos, ranges=DEFAULT_RANGES, env_assets='', extra_contact='',
                  extra_actuators='', timestep=0.05, pad_friction=1.2, pad_ct=1, pad_ca=0, grip_force=40,
                  option_extra='', pad_z=(-0.045, -0.025), palm_spheres=False, grip_lo=0.0, pad_x=(0.0,), pad_solref=None,
                  coupled=False):
  """Floating parallel gripper.  Jaw opening = 2 * q_jaw (q_jaw in [0, GRIP_STROKE]); control =
  commanded half-opening (0 = closed).  Pads: spheres r=6 mm at fingertip z = -0.045..-0.015
  below the palm, inner faces at y = +-q_jaw.  Contact budget: by default two spheres per pad
  (pad_z, resists pivoting about the jaw axis) and no palm spheres."""
  pr = 0.006
  def pad(side):   # side = +1 (left, +y) / -1 (right)
    s = '' if side > 0 else 'r'
    j = 'jl' if side > 0 else 'jr'
    ax = '0 1 0' if side > 0 else '0 -1 0'
    # priority: the pad's own solref is used for its contacts (otherwise MuJoCo averages it with the other geom's)
    sr = f' solref="{pad_solref[0]} {pad_solref[1]}" priority="1"' if pad_solref is not None else ''
    sph = '\n'.join(
        f'        <geom name="pad{s}{i}" type="sphere" size="{pr}" pos="{x:.4f} {side * pr:.4f} {z:.4f}" mass="0" '
        f'contype="{pad_ct}" conaffinity="{pad_ca}" friction="{pad_friction}"{sr} rgba="0.9 0.6 0.1 1"/>'
        for i, (z, x) in enumerate((z, x) for z in pad_z for x in pad_x))
    return f"""      <body name="jaw{s or 'l'}" gravcomp="1">
        <joint name="{j}" type="slide" axis="{ax}" range="0 {GRIP_STROKE}" armature="0.5" damping="0"/>
        <geom name="finger{s or 'l'}" type="box" size="0.011 0.004 0.03" pos="0 {side * (pr + 0.004):.4f} -0.025" mass="0.05"
              contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
{sph}
      </body>"""
  palm_c = (f'      <geom name="palm_c1" type="sphere" size="0.01" pos="0 0.03 -0.002" mass="0" contype="{pad_ct}" conaffinity="{pad_ca}" rgba="0.3 0.3 0.3 1"/>\n'
            f'      <geom name="palm_c2" type="sphere" size="0.01" pos="0 -0.03 -0.002" mass="0" contype="{pad_ct}" conaffinity="{pad_ca}" rgba="0.3 0.3 0.3 1"/>')
  # Jaws: independent by default (one position servo each).  The coupled version (joint equality jr = jl, one
  # actuator) is what a Robotiq does, but the lifted contact law that PLF-V / LFDDP plan with contains no equality
  # constraints or joint limits: there the right jaw is a free unactuated slider and no pinch exists (2026-11-24).
  eq_block = (f'  <equality>\n    <joint joint1="jr" joint2="jl" polycoef="0 1 0 0 0"/>\n  </equality>\n' if coupled else '')
  grip_r = ('' if coupled else
            f'    <position name="a_grip_r" joint="jr" kp="400" kv="40" ctrlrange="{grip_lo} {GRIP_STROKE}" forcelimited="true" '
            f'forcerange="{-grip_force} {grip_force}"/>')
  xml = f"""<mujoco model="gripper bench">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" iterations="40" tolerance="1e-6" {option_extra}/>
  <default><geom condim="3" friction="0.7"/></default>
  <asset>{env_assets}</asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>
{env_world}
{_wrist_block(wrist_pos, ranges, slide_arm=0.3, hinge_arm=0.01)}
      <geom name="palm" type="box" size="0.012 0.05 0.01" pos="0 0 0" mass="0.4" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
{palm_c if palm_spheres else ''}
{pad(+1)}
{pad(-1)}
    </body>
  </worldbody>
  <contact>
    <exclude body1="wrist" body2="jawl"/>
    <exclude body1="wrist" body2="jawr"/>
    <exclude body1="jawl" body2="jawr"/>
{extra_contact}
  </contact>
{eq_block}  <actuator>
{_wrist_actuators(ranges, 600, 60, 60, 10, 0.8, 5)}
    <position name="a_grip" joint="jl" kp="400" kv="40" ctrlrange="{grip_lo} {GRIP_STROKE}" forcelimited="true" forcerange="{-grip_force} {grip_force}"/>
{grip_r}
{extra_actuators}
  </actuator>
</mujoco>
"""
  return xml


def gripper_model(env_world, **kw):
  xml = gripper_scene(env_world, **kw)
  if not table_contacts.enabled():
    return mujoco.MjModel.from_xml_string(xml)
  return table_contacts.apply(mujoco.MjSpec.from_string(xml)).compile()
