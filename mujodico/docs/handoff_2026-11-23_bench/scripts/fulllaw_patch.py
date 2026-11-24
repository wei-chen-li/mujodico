"""Process-local patch (v2): lift EVERY constraint force (contacts + joint limits, equalities, friction loss).

The package's lifted_shooting_transcription (trajopt/_src/transcriptions.py) applies only the lifted contact forces in
its dynamics, so joint limits and equality constraints are absent from the model PLF-V's lifted IP and LFDDP plan with
(its true_step applies them).  importing this module and calling apply() replaces trajopt.lifted_shooting_transcription
(and the transcriptions module attribute) by a copy whose lifted force vector also carries the non-contact
constraint forces (consistency + dynamics), so the lifted law equals the full law at consistency.  v1 (taking the
non-contact forces from a second solve) is kept in fulllaw_patch_v1_noncontact_from_full_solve.py: it mixed two solves
and did not reproduce the true law where a contact and a limit act on the same body (pen pins on their stops).  Use with IPPAS_EXPORT='' (the on-disk export cache is keyed on the package source, not this patch).
"""
import jax
import jax.numpy as jp
import numpy as np
from mujodico import mjx
from mujodico.trajopt._src import transcriptions as _tr
from mujodico.trajopt._src.transcriptions import (_extract_contact_indices, _Contact, _step_given_efc_force,
                                                  _check_u_bounds)
import mujoco
from mujodico.trajopt._src.types import Bounds

FACE_FF = None


def lifted_shooting_transcription(
    mjmodel: mujoco.MjModel,
    contact_approximation: mjx.ContactApproxType,
    contact_smoothing: float,
    u_bounds: Bounds | None = None,
    sap_warmstart: bool = False,
    far_field_margin: float | None = None,
    lookahead_ref: float | None = None,
):
  """Builds dynamics and constraints with contact forces as lifted controls.

  The returned decision input ``w`` is ``[u, cf]``, where ``u`` is the
  actuator control and ``cf`` contains contact forces. The stage equality
  enforces that ``cf`` matches the forces computed by the forward dynamics.
  The stage inequality enforces the elliptic friction cone for each elliptic
  contact.

  Args:
    mjmodel: MuJoCo model to transcribe.
    contact_approximation: contact approximation used by the dynamics.
    contact_smoothing: contact smoothing parameter used by the dynamics.
    u_bounds: optional pair of lower and upper actuator control bounds.
              Defaults to unbounded controls.
    sap_warmstart: start the contact solve of ``stage_equality(x, w, x_next)``
              from the velocity of ``x_next`` (the lifted trajectory's next
              state) instead of the current velocity.  Same solution up to
              the solver tolerance, far fewer Newton iterations.
    far_field_margin: Hunt-Crossley only.  If given (>= 0), the normal law
              uses velocity only within this distance of contact, so the
              smoothing tail at a distance is a function of position alone
              (see ``solver_sap_util.hunt_crossley_normal_impulse``).

  Returns:
    dynamics: ``dynamics(x, w)`` returns the next state.
    stage_equality: constraint ``stage_equality(x, w[, x_next]) = 0``;
              ``stage_equality(x, w, x_next, contact_smoothing)`` evaluates
              the contact law with a runtime (traceable) smoothing.
    stage_inequality: constraint ``stage_inequality(x, w) ≥ 0``.
    w_bounds: pair of lower and upper bounds for ``w``.
  """
  model = mjx.put_model(
      mjmodel,
      impl='jax',
      contact_approximation=contact_approximation,
      contact_smoothing=contact_smoothing,
  )
  data = mjx.make_data(model, impl=model.impl)
  if sap_warmstart:
    model = model.replace(opt=model.opt.replace(
        _impl=model.opt._impl.replace(sap_warmstart_qacc=True)))
  if lookahead_ref is not None:
    model = model.replace(opt=model.opt.replace(
        _impl=model.opt._impl.replace(
            contact_lookahead_ref=float(lookahead_ref))))
  if far_field_margin is not None:
    model = model.replace(opt=model.opt.replace(
        _impl=model.opt._impl.replace(
            contact_far_field_margin=float(far_field_margin))))

  cf_addresses, contacts = _extract_contact_indices(model)
  elliptic_contacts = [c for c in contacts if c.type == _Contact.Type.ELLIPTIC]

  nu = model.nu
  ncf0 = int(cf_addresses.size)
  ff = None
  if _tr.FACE_FF:
    from mujodico.trajopt._src import face_ff as _face_ff
    ff = _face_ff.build(model, data, mjmodel, **_tr.FACE_FF)
  n_ext = 0 if ff is None else ff.n
  _nc = np.ones(int(data._impl.nefc), bool); _nc[np.asarray(cf_addresses)] = False
  nc_addresses = np.nonzero(_nc)[0]
  n_nc = int(nc_addresses.size)
  ncf = ncf0 + n_ext + n_nc
  base_smoothing = float(contact_smoothing)

  def step1(x: jax.Array, u: jax.Array, x_next=None,
            contact_smoothing=None) -> jax.Array:
    nq, nv = model.nq, model.nv
    m = model
    if contact_smoothing is not None:
      m = model.replace(opt=model.opt.replace(_impl=model.opt._impl.replace(
          contact_smoothing=contact_smoothing)))
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    if x_next is not None and sap_warmstart:
      v_guess = jax.lax.stop_gradient(x_next[nq:nq+nv])
      d = d.replace(qacc_warmstart=(v_guess - jax.lax.stop_gradient(
          x[nq:nq+nv])) / model.opt.timestep)
    d = mjx.forward(m, d)
    cf = d._impl.efc_force[cf_addresses]
    if ff is not None:
      sig = base_smoothing if contact_smoothing is None else contact_smoothing
      cf = jp.concatenate([cf, ff.law(x, sig)])
    if n_nc:
      cf = jp.concatenate([cf, d._impl.efc_force[nc_addresses]])
    return cf

  # PATCH v2 (handoff_2026-11-23_bench, 2026-11-24): LIFT EVERY CONSTRAINT FORCE.  The lifted vector is
  # [contact forces (ncf0) | far-field candidates (n_ext) | non-contact constraint forces (n_nc): joint/tendon limits,
  # equalities, friction loss].  step1 (the law, inside the consistency equality) returns all of them from ONE solve at
  # the runtime smoothing; step2 (dynamics) applies all of them.  With consistent forces this is exactly true_step.
  # The new entries are unbounded (no interior-point barrier); code indexing contacts by cf_index is unaffected.
  def step2(x: jax.Array, u: jax.Array, cf: jax.Array) -> jax.Array:
    efc_force = jp.zeros(data._impl.nefc)
    efc_force = efc_force.at[cf_addresses].set(cf[:ncf0])
    if n_nc:
      efc_force = efc_force.at[nc_addresses].set(cf[ncf0 + n_ext:])
    extra = None if ff is None else ff.qfrc(x, cf[ncf0:ncf0 + n_ext])
    return _step_given_efc_force(model, x, u, efc_force, extra)

  def unpack_control(w):
    assert w.size == nu + ncf
    return w[:nu], w[nu:]

  def dynamics(x: jax.Array, w: jax.Array) -> jax.Array:
    u, cf = unpack_control(w)
    return step2(x, u, cf)

  def true_step(x: jax.Array, u: jax.Array, contact_smoothing=None):
    """One step of the full constraint law: every constraint force of the
    SAP solve (contacts AND joint/tendon limits, equalities, friction loss)
    is applied, as in MuJoCo.  ``dynamics`` applies only the lifted contact
    forces; the two agree whenever no non-contact constraint is active.
    Returns (x_next, contact forces)."""
    nq, nv = model.nq, model.nv
    m = model
    if contact_smoothing is not None:
      m = model.replace(opt=model.opt.replace(_impl=model.opt._impl.replace(
          contact_smoothing=contact_smoothing)))
    d = data.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    d = mjx.forward(m, d)
    efc = d._impl.efc_force
    cf = efc[cf_addresses]
    if ff is not None:          # the far-field candidates are planning-only: zero under the true law
      cf = jp.concatenate([cf, jp.zeros(n_ext)])
    if n_nc:
      cf = jp.concatenate([cf, efc[nc_addresses]])
    return _step_given_efc_force(model, x, u, efc), cf

  dynamics.true_step = true_step

  def stage_equality(x: jax.Array, w: jax.Array, x_next=None,
                     contact_smoothing=None) -> jax.Array:
    u, cf = unpack_control(w)
    return cf - step1(x, u, x_next, contact_smoothing)

  def stage_inequality(x: jax.Array, w: jax.Array) -> jax.Array:
    del x
    _, cf = unpack_control(w)
    constraints = jp.zeros(len(elliptic_contacts))
    for i, contact in enumerate(elliptic_contacts):
      normal = cf[contact.cf_index]
      tangent = cf[contact.cf_index+1:contact.cf_index+contact.dim] / contact.friction
      constraints = constraints.at[i].set(normal * normal - tangent.dot(tangent))
    return constraints

  cf_lb = np.zeros(ncf)
  cf_ub = np.full(ncf, np.inf)
  for contact in elliptic_contacts:
    cf_lb[contact.cf_index+1:contact.cf_index+contact.dim] = -np.inf
  # far-field candidates: no bound, hence no interior-point barrier (the consistency equality alone keeps
  # them at their law value >= 0); a barrier would reward raising every face's tail = pull toward contact
  cf_lb[ncf0:] = -np.inf          # far-field candidates and (patch v2) non-contact constraint forces: unbounded

  u_lb, u_ub = _check_u_bounds(u_bounds, nu)
  w_bounds = (np.concatenate([u_lb, cf_lb]), np.concatenate([u_ub, cf_ub]))

  return dynamics, stage_equality, stage_inequality, w_bounds




def apply():
  import os
  from mujodico import trajopt
  assert os.environ.get('IPPAS_EXPORT', None) == '', "set IPPAS_EXPORT='' with the full-law patch"
  _tr.lifted_shooting_transcription = lifted_shooting_transcription
  trajopt.lifted_shooting_transcription = lifted_shooting_transcription
  import mujodico.examples.methods.transcription_lifted_multiple_shooting as _lms
  if hasattr(_lms, 'trajopt'):
    _lms.trajopt.lifted_shooting_transcription = lifted_shooting_transcription
