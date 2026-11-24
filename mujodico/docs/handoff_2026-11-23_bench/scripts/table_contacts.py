"""Orientation-complete table contacts (2026-11-24).

Before this fix every free object touched the table only through helper spheres placed for its START
orientation; a tossed / flipped object sank into the table (PLF-V's AllegroEraserWipe "solve" tossed the
eraser, which came to rest upside down with handle and felt block 25 mm inside the tabletop).

Fix, applied to every bench model at compile time (embodiments.compile_xml):
  * the table box (top face z = 0) becomes a PLANE at z = 0 (same surface; MJX plane-box = 4 contacts,
    plane-capsule 2, plane-cylinder 3, plane-sphere 1 -- valid in every orientation);
  * ADD: the object's real collision geometry gets a dedicated contype bit that only the table has in its
    conaffinity, so it collides with the table and with nothing new;
  * DROP: helper spheres whose job was table support lose their table contact (only that pair: a table
    conaffinity bit used solely by DROP geoms is cleared; otherwise the geom's table-matching bits are).
Helpers that are physically meaningful in every orientation stay: spheres that are exactly a capsule's end
caps (pen, brush handle), knobs / handle-end spheres larger than the handle (hammer, screwdriver, spatula),
the toast's mid-plane spheres, and the hidden stand-off feet that hold the eraser / brush head off the table.

BENCH_TABLE=legacy restores the old models (no transform).
"""
import os
import mujoco

TABLE_BIT = 1 << 20

ADD = {
    # gripper tasks
    'cubeA', 'cubeB',                 # replaces the 4 bottom-corner spheres of each cube (corners still hit cube B)
    'peg',                            # replaces peg_e1/e2 (spheres inscribed in the box ends)
    'leg',                            # replaces leg_e2 (inscribed in the box end); leg_e1 = the stud's end cap, kept
    'drawer',                         # replaces 6 of the drawer's 8 corner spheres
    # Allegro tool tasks
    'ham_head', 'ham_handle', 'ham_knob',   # replaces ham_s1/s2 (under the head)
    'sd_blade', 'sd_s1', 'sd_s2',           # replaces sd_s3 (blade centre); s1/s2 = handle-end spheres (r > handle)
    'sp_handle', 'sp_blade', 'sp_knob',     # replaces sp_b1/b2 (two blade corners)
    'pen_tip',                              # pen_sup0/1 are exactly the barrel capsule's end caps: kept
    'br_head',                              # br_sup0/1 = handle end caps, br_sup2 = stand-off foot: kept
    'er_handle', 'er_block',                # er_sup0-2 = stand-off feet: kept
}
DROP = {
    'cubeA_c-1-1-1', 'cubeA_c-1+1-1', 'cubeA_c+1-1-1', 'cubeA_c+1+1-1',
    'cubeB_c-1-1-1', 'cubeB_c-1+1-1', 'cubeB_c+1-1-1', 'cubeB_c+1+1-1',
    'peg_e1', 'peg_e2', 'leg_e2',
    'drw_c-1-1-1', 'drw_c-1-1+1', 'drw_c-1+1-1', 'drw_c+1-1-1', 'drw_c+1-1+1', 'drw_c+1+1-1',
    'ham_s1', 'ham_s2', 'sd_s3', 'sp_b1', 'sp_b2',
}
PRIORITY_FRICTION = {'br_head': 0.25}


def enabled():
  return os.environ.get('BENCH_TABLE', 'plane') != 'legacy'


# --- plane-box collider with true distances (process-local patch of MJX's collider table) -------------------
# MJX's plane_convex reports only the vertices within 1 mm of the deepest one and marks the others invalid
# (dist = 1).  The contact laws here need every listed contact with its TRUE signed distance (like the sphere
# and capsule colliders give): with dt 0.05 a slightly tilted box otherwise loses its far-end support for a
# whole step and rotates through the plane (peg: 78 mm in 90 steps of zero control).  This collider returns
# the 4 lowest of the box's 8 vertices with their true distances.  Only (PLANE, BOX) is replaced.
def _install_plane_box():
  import jax
  from jax import numpy as jp
  from mujodico.mjx._src import math as mmath
  from mujodico.mjx._src import collision_driver as CD
  from mujodico.mjx._src import collision_convex as CC
  from mujodico.mjx._src.types import GeomType

  @CC.collider(ncon=4)
  def plane_box_true(plane, convex):
    vert = convex.vert
    plane_pos = convex.mat.T @ (plane.pos - convex.pos)
    n_loc = convex.mat.T @ plane.mat[:, 2]
    depth = (plane_pos - vert) @ n_loc           # > 0: below the plane
    _, idx = jax.lax.top_k(depth, 4)              # the 4 lowest vertices
    n = plane.mat[:, 2]
    dist = -depth[idx]
    pos = convex.pos + vert[idx] @ convex.mat.T
    pos = pos - 0.5 * dist[:, None] * n
    frame = jp.stack([mmath.make_frame(n)] * 4, axis=0)
    return dist, pos, frame

  CD._COLLISION_FUNC[(GeomType.PLANE, GeomType.BOX)] = plane_box_true


if enabled():
  _install_plane_box()


def apply(spec):
  """Edit an MjSpec in place (see module docstring).  No-op when the spec has no geom named 'table'."""
  geoms = {g.name: g for g in spec.geoms}
  tb = geoms.get('table')
  if tb is None:
    return spec
  drop = [g for n, g in geoms.items() if n in DROP]
  # 1. table conaffinity bits used only by DROP geoms -> clear (keeps the DROP geoms' other partners)
  for b in range(32):
    bit = 1 << b
    if tb.conaffinity & bit:
      users = [g for n, g in geoms.items() if n != 'table' and g.contype & bit]
      if users and all(g.name in DROP for g in users):
        tb.conaffinity &= ~bit
  # 2. remaining DROP geoms: remove only their table-matching bits
  for g in drop:
    g.contype &= ~tb.conaffinity
    g.conaffinity &= ~tb.contype
  # 3. ADD geoms collide with the table through a private bit
  for n, g in geoms.items():
    if n in ADD:
      g.contype |= TABLE_BIT
  tb.conaffinity |= TABLE_BIT
  # 3b. per-geom friction that overrides the max-combination (priority 1): the brush head is bristles, which
  #     slide on the table; as a rigid box with the table's 0.6 its corner snagged at touch-down, the brush turned
  #     in the grip and the head rode over a debris piece (witness 0.65 -> 0.0003-0.0009 with 0.25, seeds 0-2)
  for n, mu in PRIORITY_FRICTION.items():
    if n in geoms:
      geoms[n].priority = 1
      geoms[n].friction = [mu, 0.005, 0.0001]
  # 4. table box (top at z = 0) -> plane at z = 0
  top = tb.pos[2] + tb.size[2]
  tb.type = mujoco.mjtGeom.mjGEOM_PLANE
  tb.pos = [tb.pos[0], tb.pos[1], top]
  tb.size = [tb.size[0], tb.size[1], 0.02]
  return spec
