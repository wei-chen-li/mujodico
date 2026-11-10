"""Dense cross-checks of the structured IP-PAS primal-dual solves.

The full primal-dual Newton system of FATROP's original and restoration NLPs
is assembled densely (autodiff of the Lagrangian) and compared with the
structured reduce -> Riccati/PAS -> dereduce path of ``IpPasSolver``.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from mujodico.trajopt._src.fatrop_ref import ippas  # noqa: E402

N, NX, NW = 5, 3, 2


def _problem(backend='pas', hessian_mode='exact_lagrangian'):
  def dyn(x, w):
    return jnp.array([x[0] + 0.1 * x[1] + 0.05 * jnp.sin(x[2]),
                      x[1] + 0.1 * w[0] - 0.2 * jnp.sin(x[0]),
                      x[2] + 0.1 * w[1] * x[1]])

  def eq(x, w):
    return jnp.array([w[0] - jnp.tanh(x[0] + w[1])])

  def ineq(x, w):
    return jnp.array([x[0] + x[1] ** 2, w[1], x[2] * w[0]])

  cost = lambda x, w, p: jnp.sum((x - p['g']) ** 2) + 0.3 * jnp.sum(w ** 2) + \
      0.1 * x[0] * w[1]
  term = lambda x, p: 5.0 * jnp.sum((x - p['g']) ** 2)
  lb = np.array([-1.0, -0.7, -np.inf])
  ub = np.array([np.inf, 0.8, 2.0])
  return ippas.IpPasSolver(
      horizon=N, nx=NX, nw=NW, stage_cost=cost, terminal_cost=term,
      dynamics=dyn, stage_equality=eq, stage_inequality=ineq, ineq_lb=lb,
      ineq_ub=ub, hessian_mode=hessian_mode, linear_backend=backend,
      options=ippas.IpPasOptions(max_iter=20, resto_relax_initial=True))


def _random_state(sol, mode, seed=0):
  rng = np.random.default_rng(seed)
  r = lambda *s: jnp.asarray(rng.normal(size=s))
  pos = lambda *s: jnp.asarray(rng.uniform(0.2, 2.0, size=s))
  ne, ni = sol.ne, sol.ni
  x = r(N + 1, NX) * 0.5
  w = r(N, NW) * 0.5
  lb, ub = sol._lb, sol._ub
  s = np.asarray(rng.uniform(-0.5, 0.5, size=(N, ni)))
  s = np.clip(s, np.where(np.isfinite(lb), lb + 0.1, -9), np.where(
      np.isfinite(ub), ub - 0.1, 9))
  y = ippas.Rows(r(NX), r(N, NX), r(N, ne), r(N, ni))
  flo, fhi = sol._flo, sol._fhi
  zL = pos(N, ni) * flo
  zU = pos(N, ni) * fhi
  e = lambda: ippas.ERows(pos(NX), pos(N, ne), pos(N, ni))
  it = ippas.Iterate(x, w, jnp.asarray(s), y, zL, zU, e(), e(), e(), e(),
                     jnp.asarray(0.05))
  params = {'x0': r(NX), 'g': r(NX)}
  refs = ippas.Refs(x + 0.1, w - 0.1, jnp.asarray(s) + 0.05,
                    jnp.asarray(0.3), it, None, None, jnp.asarray(1.0), it.y)
  return it, params, refs


def _dense_direction(sol, it, params, refs, mode, delta_w):
  """Newton step of the full primal-dual system built by autodiff."""
  ne, ni = sol.ne, sol.ni
  flo = jnp.asarray(sol._flo)[None] * jnp.ones((N, ni), bool)
  fhi = jnp.asarray(sol._fhi)[None] * jnp.ones((N, ni), bool)
  lb = jnp.asarray(np.where(sol._flo, sol._lb, 0.0))[None]
  ub = jnp.asarray(np.where(sol._fhi, sol._ub, 0.0))[None]
  r = mode == 1
  shapes = dict(x=(N + 1, NX), w=(N, NW), s=(N, ni))
  e_shapes = [(NX,), (N, ne), (N, ni)]

  def unflat_primal(v):
    o = 0
    out = {}
    for k, sh in shapes.items():
      n = int(np.prod(sh))
      out[k] = v[o:o + n].reshape(sh)
      o += n
    p = []
    for sh in e_shapes:
      n = int(np.prod(sh))
      p.append(v[o:o + n].reshape(sh))
      o += n
    q = []
    for sh in e_shapes:
      n = int(np.prod(sh))
      q.append(v[o:o + n].reshape(sh))
      o += n
    return out, ippas.ERows(*p), ippas.ERows(*q)

  def flat(*trees):
    return jnp.concatenate([jnp.ravel(l) for t in trees
                            for l in jax.tree.leaves(t)])

  v0 = flat((it.x, it.w, it.s), it.p, it.n)

  def constraints(v):
    pr, p, n = unflat_primal(v)
    vals = sol._values(pr['x'], pr['w'], pr['s'], params)
    c = vals.c
    if r:
      c = ippas.Rows(c.c0 + p.c0 - n.c0, c.cd, c.ch + p.ch - n.ch,
                     c.cg + p.cg - n.cg)
    return flat(c)

  def objective(v):
    pr, p, n = unflat_primal(v)
    if r:
      dx = (pr['x'] - refs.xref) / jnp.maximum(1.0, jnp.abs(refs.xref))
      dw = (pr['w'] - refs.wref) / jnp.maximum(1.0, jnp.abs(refs.wref))
      ds = (pr['s'] - refs.sref) / jnp.maximum(1.0, jnp.abs(refs.sref))
      return 0.5 * refs.zeta * (jnp.sum(dx**2) + jnp.sum(dw**2) +
                                jnp.sum(ds**2)) + sol.opt.resto_rho * (
                                    ippas._sum(p) + ippas._sum(n))
    return sol._values(pr['x'], pr['w'], pr['s'], params).obj

  y0 = flat(it.y)

  nv = v0.size
  ny = y0.size
  grad_obj = jax.grad(objective)(v0)
  J = jax.jacfwd(constraints)(v0)
  if sol.hessian_mode == 'exact_lagrangian':
    H = jax.hessian(lambda v: (0.0 if r else 1.0) * objective(v) +
                    jnp.dot(y0, constraints(v)))(v0)
  else:
    H = jax.hessian(lambda v: (0.0 if r else 1.0) * objective(v))(v0)
  # Primal damping (resto proximity) + delta_w on x, w, s, p, n.
  npr = sum(int(np.prod(sh)) for sh in shapes.values())
  damp = jnp.zeros(nv)
  if r:
    dr = flat((1.0 / jnp.maximum(1.0, jnp.abs(refs.xref)),
               1.0 / jnp.maximum(1.0, jnp.abs(refs.wref)),
               1.0 / jnp.maximum(1.0, jnp.abs(refs.sref))))
    damp = damp.at[:npr].set(refs.zeta * dr ** 2)
  Hd = H + jnp.diag(damp + delta_w)
  # bounds: on s (orig) and on p, n >= 0 (resto)
  s_idx = np.arange(npr - N * ni, npr)
  Sl = (it.s - lb).ravel()
  Su = (ub - it.s).ravel()
  bl_idx, bl_d, bl_z = [s_idx[flo.ravel()]], [Sl[flo.ravel()]], [
      it.zL.ravel()[flo.ravel()]]
  bu_idx, bu_d, bu_z = [s_idx[fhi.ravel()]], [Su[fhi.ravel()]], [
      it.zU.ravel()[fhi.ravel()]]
  if r:
    pn_idx = np.arange(npr, nv)
    bl_idx.append(pn_idx)
    bl_d.append(flat(it.p, it.n))
    bl_z.append(flat(it.zp, it.zn))
  bl_idx = np.concatenate(bl_idx)
  bl_d = jnp.concatenate(bl_d)
  bl_z = jnp.concatenate(bl_z)
  bu_idx = np.concatenate(bu_idx)
  bu_d = jnp.concatenate(bu_d)
  bu_z = jnp.concatenate(bu_z)
  nl, nu_ = bl_idx.size, bu_idx.size
  mu = it.mu
  # Unknowns: [dv (nv), dy (ny), dzl (nl), dzu (nu)]
  K = jnp.zeros((nv + ny + nl + nu_,) * 2)
  K = K.at[:nv, :nv].set(Hd)
  K = K.at[:nv, nv:nv + ny].set(J.T)
  K = K.at[nv:nv + ny, :nv].set(J)
  Il = jnp.zeros((nv, nl)).at[bl_idx, jnp.arange(nl)].set(1.0)
  Iu = jnp.zeros((nv, nu_)).at[bu_idx, jnp.arange(nu_)].set(1.0)
  K = K.at[:nv, nv + ny:nv + ny + nl].set(-Il)
  K = K.at[:nv, nv + ny + nl:].set(Iu)
  K = K.at[nv + ny:nv + ny + nl, :nv].set(bl_z[:, None] * Il.T)
  K = K.at[nv + ny:nv + ny + nl, nv + ny:nv + ny + nl].set(jnp.diag(bl_d))
  K = K.at[nv + ny + nl:, :nv].set(-bu_z[:, None] * Iu.T)
  K = K.at[nv + ny + nl:, nv + ny + nl:].set(jnp.diag(bu_d))
  zl_full = Il @ bl_z
  zu_full = Iu @ bu_z
  rhs = jnp.concatenate([
      grad_obj + J.T @ y0 - zl_full + zu_full,
      constraints(v0),
      bl_d * bl_z - mu,
      bu_d * bu_z - mu])
  sol_d = jnp.linalg.solve(K, -rhs)
  dv = sol_d[:nv]
  dy = sol_d[nv:nv + ny]
  pr, dp, dn = unflat_primal(dv)
  return pr, dp, dn, dy


def _structured_direction(sol, it, params, refs, mode, delta_w):
  lin = sol._linearize(it, params)
  sys_, red = sol._assemble(it, lin, mode, refs, delta_w, False)
  b = sol._full_rhs(it, lin, mode, refs)
  d, flag, _ = sol._solve_full(sys_, red, b, factor=True)
  return d, flag


def _check(backend, hessian_mode, mode):
  sol = _problem(backend, hessian_mode)
  it, params, refs = _random_state(sol, mode)
  dw = 0.5  # keeps the random system definite
  d, flag = _structured_direction(sol, it, params, refs, mode, dw)
  pr, dp, dn, dy = _dense_direction(sol, it, params, refs, mode, dw)
  np.testing.assert_allclose(d.dx, pr['x'], rtol=1e-8, atol=1e-9)
  np.testing.assert_allclose(d.dw, pr['w'], rtol=1e-8, atol=1e-9)
  np.testing.assert_allclose(d.ds, pr['s'], rtol=1e-8, atol=1e-9)
  dyf = jnp.concatenate([jnp.ravel(l) for l in jax.tree.leaves(d.dy)])
  np.testing.assert_allclose(dyf, dy, rtol=1e-8, atol=1e-9)
  if mode == 1:
    for a, b in zip(jax.tree.leaves(d.dp), jax.tree.leaves(dp)):
      np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
    for a, b in zip(jax.tree.leaves(d.dn), jax.tree.leaves(dn)):
      np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)


def test_original_pas_exact():
  _check('pas', 'exact_lagrangian', 0)


def test_original_serial_cost():
  _check('serial', 'cost', 0)


def test_restoration_pas_exact():
  _check('pas', 'exact_lagrangian', 1)


def test_restoration_serial_exact():
  _check('serial', 'exact_lagrangian', 1)


if __name__ == '__main__':
  for name, fn in list(globals().items()):
    if name.startswith('test_'):
      try:
        fn()
        print('PASS', name)
      except AssertionError as e:
        print('FAIL', name, str(e)[:2000])
