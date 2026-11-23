"""LPEC branch step with a structured (parallel-in-time) QP solver instead of Gurobi's sparse QP.

Same linearization and constraints as lpec_step.py (about the incumbent's TRUE trajectory; biactive pairs
(t, j): gap < GAP_TOL and normal impulse < F_TOL).  With the impulse step eliminated through the linearized law,
dgamma_t = Phi_x dx_t + Phi_u du_t + s_t, the step is an optimal-control QP in (dx_t, dw_t), dw_t = (du_t, s_t):

    min   sum_t 1/2 dx' Hxx dx + lx' dx + 1/2 du' Huu du + lu' du  +  1/2 dx_T' HT dx_T + lT' dx_T
    s.t.  dx_0 = 0,  dx_{t+1} = A_t dx_t + B_t dw_t + d_t,   A = f_x + f_g Phi_x,  B = [f_u + f_g Phi_u, f_g M_t]
          |du_t|_inf <= Delta (within the actuator box),  |dx_t|_inf <= 0.5
          pressed edges (z = 1):  0 <= s <= M_F;   min(phi,0) <= phi + grad phi' dx <= 0      (gap stays closed)
          separated pairs (z = 0): s = 0 (M_t masks the column);  min(phi,0) <= phi + grad phi' dx <= M_G

It is solved by a primal-dual interior-point method (Mehrotra predictor-corrector) whose Newton systems are the
stage-structured KKT systems of the package (trajopt/_src/fatrop_ref/kkt.py), factored by a parallel associative
scan over time (O(log T) depth) on the GPU.  The whole solve is one jitted program, vmapped over branch sets.

Branch choice (mode):
  milp   Gurobi MILP with the linear objective chooses z (as lpec_step.py); the structured QP gives the step
  sign   no integer program: solve the QP with every biactive pair both closed and unloaded (s = 0, gap closed) and
         read the multipliers: pressing pays if dL/ds < 0 on an edge (first-order decrease M_F |dL/ds|), separating
         pays if the multiplier of the closed-gap bound is positive (decrease M_G y); take the larger per pair
  local  the sign-test set and every single-pair flip of it, solved as one batched QP; keep the lowest model cost"""
import os, time
import numpy as np
import jax, jax.numpy as jnp
from mujodico.examples.methods import util
from mujodico.trajopt._src import transcriptions as trx
from mujodico.trajopt._src.fatrop_ref import kkt

GAP_TOL, F_TOL, M_F, M_G = 1e-3, 1e-6, 5.0, 0.05
BIG = 1e3          # bound of a free row (inactive)
EQW = 1e-5         # half-width of an equality written as a band (closed gap with phi >= 0): 10 micrometres


def _psd(H):
  w, V = jnp.linalg.eigh(0.5 * (H + jnp.swapaxes(H, -1, -2)))
  return (V * jnp.maximum(w, 0.0)[..., None, :]) @ jnp.swapaxes(V, -1, -2)


def qp_ipm(Qx, qx, Rw, rw, QN, qN, A, B, a, Gx, Gw, lo, hi, iters=60):
  """min sum_t 1/2 x'Qx x + qx'x + 1/2 w'Rw w + rw'w + 1/2 xN'QN xN + qN'xN
     s.t. x_0 = 0, x_{t+1} = A x_t + B w_t + a_t, lo <= Gx x_t + Gw w_t <= hi.
  Returns (x (T+1,nx), w (T,nw), lam (T,nx), y_lo, y_hi (T,m0) >= 0 multipliers of the two sides, objective, info)."""
  T, nx = qx.shape
  nw = rw.shape[1]
  G2x = jnp.concatenate([Gx, -Gx], 1); G2w = jnp.concatenate([Gw, -Gw], 1)
  g0 = jnp.concatenate([-lo, hi], 1)                      # rows h = G2 v + g0 >= 0
  m = g0.shape[1]
  Z0x = jnp.zeros((T, 0, nx)); Z0w = jnp.zeros((T, 0, nw)); Z0 = jnp.zeros((T, 0))
  regx = jnp.full((T, nx), 1e-9); regw = jnp.full((T, nw), 1e-9); regf = jnp.full((nx,), 1e-9)

  def hval(x, w):
    return jnp.einsum('tmx,tx->tm', G2x, x[:-1]) + jnp.einsum('tmw,tw->tm', G2w, w) + g0

  def resid(x, w, lam, lam0, t, y):
    gam = -y
    q = (jnp.einsum('tij,tj->ti', Qx, x[:-1]) + qx + jnp.einsum('tji,tj->ti', A, lam)
         + jnp.einsum('tmx,tm->tx', G2x, gam))
    q = q.at[1:].add(-lam[:-1]).at[0].add(lam0)
    r = jnp.einsum('tij,tj->ti', Rw, w) + rw + jnp.einsum('tji,tj->ti', B, lam) + jnp.einsum('tmw,tm->tw', G2w, gam)
    qf = QN @ x[-1] + qN - lam[-1]
    dyn = jnp.einsum('tij,tj->ti', A, x[:-1]) + jnp.einsum('tij,tj->ti', B, w) + a - x[1:]
    rp = hval(x, w) - t
    return q, r, qf, dyn, rp

  def newton(x, w, lam, lam0, t, y, rc):
    q, r, qf, dyn, rp = resid(x, w, lam, lam0, t, y)
    sysk = kkt.KKTSystem(Q=Qx, S=jnp.zeros((T, nx, nw)), R=Rw, Qf=QN, Dxx=regx, Dxw=regw, Dxf=regf, A=A, B=B,
                         E=Z0x, F=Z0w, Gx=G2x, Gw=G2w, De=Z0, Dii=t / y, De0=jnp.zeros(nx), resto=jnp.asarray(False))
    rhs = kkt.KKTRhs(q=q, r=r, qf=qf, a=dyn, e=Z0, g=rp - rc / y, c0=x[0])
    sol, _ = kkt.solve_pas(sysk, rhs)
    dgam = sol.dgamma
    dt = (rc + t * dgam) / y
    return sol.dx, sol.dw, sol.dlam, sol.dlam0, dt, -dgam

  def ftb(v, dv):
    r_ = jnp.where(dv < 0, -0.995 * v / jnp.where(dv < 0, dv, -1.0), jnp.inf)
    return jnp.minimum(1.0, jnp.min(r_))

  x = jnp.zeros((T + 1, nx)); w = jnp.zeros((T, nw)); lam = jnp.zeros((T, nx)); lam0 = jnp.zeros(nx)
  t = jnp.maximum(hval(x, w), 1.0); y = jnp.ones((T, m))

  def step_(st):
    x, w, lam, lam0, t, y = st
    mu = jnp.mean(t * y)
    dx, dw, dl, dl0, dt, dy = newton(x, w, lam, lam0, t, y, -t * y)               # predictor (affine)
    ap, ad = ftb(t, dt), ftb(y, dy)
    mu_a = jnp.mean((t + ap * dt) * (y + ad * dy))
    sig = (mu_a / mu) ** 3
    dx, dw, dl, dl0, dt, dy = newton(x, w, lam, lam0, t, y, sig * mu - t * y - dt * dy)   # corrector
    ap, ad = ftb(t, dt), ftb(y, dy)
    return (x + ap * dx, w + ap * dw, lam + ad * dl, lam0 + ad * dl0, jnp.maximum(t + ap * dt, 1e-14),
            jnp.maximum(y + ad * dy, 1e-14))

  def body(_, st):   # converged (complementarity and residuals at roundoff): stop, before t/y underflow
    q, r, qf, dyn, rp = resid(*st)
    done = ((jnp.mean(st[4] * st[5]) < 1e-10) & (jnp.max(jnp.abs(dyn)) < 1e-6) & (jnp.max(jnp.abs(rp)) < 1e-6))
    return jax.lax.cond(done, lambda s_: s_, step_, st)

  x, w, lam, lam0, t, y = jax.lax.fori_loop(0, iters, body, (x, w, lam, lam0, t, y))
  q, r, qf, dyn, rp = resid(x, w, lam, lam0, t, y)
  obj = (0.5 * jnp.einsum('ti,tij,tj->', x[:-1], Qx, x[:-1]) + jnp.sum(qx * x[:-1]) + 0.5 * jnp.einsum('ti,tij,tj->', w, Rw, w)
         + jnp.sum(rw * w) + 0.5 * x[-1] @ QN @ x[-1] + qN @ x[-1])
  m0 = m // 2
  info = dict(mu=jnp.mean(t * y), dyn=jnp.max(jnp.abs(dyn)), prim=jnp.max(jnp.abs(rp)),
              dual=jnp.maximum(jnp.max(jnp.abs(q)), jnp.max(jnp.abs(r))))
  return x, w, lam, y[:, :m0], y[:, m0:], obj, info


def make_lpec_fast(task, S, truem, lms):
  trans = lms.transcription
  nu, ncf, T, nx, nq = S.nu, S.ncf, S.N, task.nx, task.model.nq
  mm = trx.mjx.put_model(task.model, impl='jax', contact_approximation=task.contact_approx, contact_smoothing=task.contact_smoothing)
  cf_addr, contacts = trx._extract_contact_indices(mm)
  d0 = trx.mjx.make_data(mm, impl=mm.impl)
  first = np.array([c.cf_index for c in contacts]); nc = len(contacts)
  edges = [list(range(c.cf_index, c.cf_index + (2 * (c.dim - 1) if c.type == trx._Contact.Type.PYRAMIDAL else 1))) for c in contacts]
  E2C = np.zeros((ncf, nc));                                    # edge -> contact incidence
  for j, es in enumerate(edges):
    E2C[es, j] = 1.0
  E2C = jnp.asarray(E2C)
  gapf = lambda x: trx.mjx.fwd_position(mm, d0.replace(qpos=x[:nq]))._impl.efc_pos[cf_addr][first]
  ts = truem.transcription.true_step
  eps = jnp.asarray(S.base * 1e-3)
  CPU = jax.devices('cpu')[0]; DEV = jax.devices()[0]
  ulb, uub = (np.asarray(b_, float) for b_ in task.u_bounds)
  Qm = np.asarray(getattr(task, 'metric_Q', task.Q))
  goal = lambda Xe: float(util.state_cost(Qm, task.x_goal, np.asarray(Xe[-1]), task.quaternion_w_idxs))
  true_cf = jax.jit(jax.vmap(lambda x, u: ts(x, u, jnp.asarray(0.0))[1]))

  def linearize(X, U, CF, P):
    def st(x, u, c, xn):
      w = jnp.concatenate([u, c])
      fv = trans.dynamics(x, w)
      fx, fw = jax.jacfwd(trans.dynamics, argnums=(0, 1))(x, w)
      Lx, Lu = jax.jacfwd(S.law, argnums=(0, 1))(x, u, eps)
      g, gx = gapf(x), jax.jacfwd(gapf)(x)
      lx, lu = jax.grad(S.lst, (0, 1))(x, u, P)
      (hxx, _), (_, huu) = jax.hessian(S.lst, (0, 1))(x, u, P)
      return fv - xn, fx, fw[:, :nu], fw[:, nu:], Lx, Lu, g, gx, lx, lu, hxx, huu
    out = jax.vmap(st)(X[:-1], U, CF, X[1:])
    return out + (jax.grad(S.lterm)(X[-1], P), jax.hessian(S.lterm)(X[-1], P))
  lin_fn = jax.jit(linearize)
  if os.environ.get('PLF_FN_EXPORT', '1') != '0' and getattr(lms.solver, 'export_key', None) and lms.solver.export_dir:
    import hashlib
    from hs_export import ExportedJit
    _k = hashlib.sha256('|'.join((lms.solver.export_key, getattr(truem.solver, 'export_key', '') or '', 'lpec_lin_v1')).encode()).hexdigest()
    lin_fn = ExportedJit(linearize, 'lpec_lin', _k, lms.solver.export_dir)

  def qp_data(L, U0, D):
    dfc, fx, fu, fc, Lx, Lu, gap, gx, lx, lu, hxx, huu, lN, hNN = L
    A = fx + fc @ Lx; Bu = fu + fc @ Lu
    Qx = _psd(hxx) + 1e-8 * jnp.eye(nx); QN = _psd(hNN) + 1e-8 * jnp.eye(nx)
    Ruu = _psd(huu) + 1e-8 * jnp.eye(nu)
    Rw = jnp.zeros((T, nu + ncf, nu + ncf)).at[:, :nu, :nu].set(Ruu).at[:, nu:, nu:].set(1e-6 * jnp.eye(ncf))
    rw = jnp.concatenate([lu, jnp.zeros((T, ncf))], 1)
    ulo = jnp.maximum(-D, ulb - U0); uhi = jnp.minimum(D, uub - U0)
    return A, Bu, fc, dfc, Qx, lx, Rw, rw, QN, lN, gap, gx, ulo, uhi

  def solve_branch(qd, bia, z, zs):
    """bia (T, nc) biactive mask; z (T, nc) pairs whose gap stays closed; zs (T, nc) pairs with extra force s allowed
    (z = zs for a branch choice; z = bia, zs = 0 for the tightened problem of the sign test)."""
    A, Bu, fc, dfc, Qx, lx, Rw, rw, QN, lN, gap, gx, ulo, uhi = qd
    press_e = (zs @ E2C.T) > 0.5                                  # (T, ncf) edges with s allowed
    B = jnp.concatenate([Bu, fc * press_e[:, None, :]], 2)
    I_u = jnp.eye(nu, nu + ncf); I_s = jnp.eye(ncf, nu + ncf, k=nu)
    # rows: du box | s box | gap rows (biactive only) | dx box
    Gw = jnp.concatenate([jnp.broadcast_to(I_u, (T, nu, nu + ncf)), jnp.broadcast_to(I_s, (T, ncf, nu + ncf)),
                          jnp.zeros((T, nc, nu + ncf)), jnp.zeros((T, nx, nu + ncf))], 1)
    Gx = jnp.concatenate([jnp.zeros((T, nu + ncf, nx)), gx, jnp.broadcast_to(jnp.eye(nx), (T, nx, nx))], 1)
    s_lo = jnp.where(press_e, 0.0, -BIG); s_hi = jnp.where(press_e, M_F, BIG)
    g_lo = -jnp.maximum(gap, 0.0); g_hi = jnp.where(z > 0.5, -gap, M_G - gap)
    g_lo, g_hi = jnp.where(bia > 0.5, g_lo, -BIG), jnp.where(bia > 0.5, g_hi, BIG)
    band = (g_hi - g_lo) < 2 * EQW                                 # closed gap with phi >= 0: equality as a band
    g_lo, g_hi = jnp.where(band, g_lo - EQW, g_lo), jnp.where(band, g_hi + EQW, g_hi)
    lo = jnp.concatenate([ulo, s_lo, g_lo, jnp.full((T, nx), -0.5)], 1)
    hi = jnp.concatenate([uhi, s_hi, g_hi, jnp.full((T, nx), 0.5)], 1)
    if os.environ.get('LPEC_FAST_DEBUG'):
      return (Qx, lx, Rw, rw, QN, lN, A, B, dfc, Gx, Gw, lo, hi)
    x, w, lam, ylo, yhi, obj, info = qp_ipm(Qx, lx, Rw, rw, QN, lN, A, B, dfc, Gx, Gw, lo, hi)
    dLds = jnp.einsum('tie,ti->te', fc, lam)                       # dL/ds on every edge (s enters x_{t+1} via f_g)
    y_gap_hi = yhi[:, nu + ncf:nu + ncf + nc]
    return w[:, :nu], obj, dLds, y_gap_hi, info

  solve_j = jax.jit(solve_branch)
  solve_batch = jax.jit(jax.vmap(lambda qd, b, z: solve_branch(qd, b, z, z), in_axes=(None, None, 0)))

  cache = {}

  def lin_at(us):
    key = np.asarray(us).tobytes()
    if key in cache:
      return cache[key]
    cache.clear()
    tb = time.perf_counter()
    usc = jax.device_put(jnp.asarray(np.asarray(us)), CPU)
    X, _ = truem._rollout(jax.device_put(jnp.asarray(task.x_0), CPU), usc, jnp.asarray(0.0))
    CF = true_cf(X[:-1], usc)
    L = lin_fn(*jax.device_put((X, usc, CF, trans.params), DEV))
    nrm = np.asarray(CF) @ np.asarray(S.G).T
    gap = np.asarray(L[6])
    bia = ((gap < GAP_TOL) & (nrm < F_TOL)).astype(float)
    out = dict(L=L, bia=bia, U0=np.asarray(us), t_lin=time.perf_counter() - tb)
    cache[key] = out
    return out

  def problem(us, D, z):   # debug: the QP data of a branch choice (LPEC_FAST_DEBUG=1)
    c = lin_at(us); qd = qp_data(jax.device_put(c['L'], DEV), jnp.asarray(c['U0']), D)
    return solve_branch(qd, jnp.asarray(c['bia']), jnp.asarray(z), jnp.asarray(z))

  def step(us, D, mode='sign', milp_z=None):
    t0 = time.perf_counter()
    c = lin_at(us)
    bia = c['bia']; nbi = int(bia.sum())
    if nbi == 0:
      return dict(nbi=0, time=time.perf_counter() - t0)
    qd = qp_data(jax.device_put(c['L'], DEV), jnp.asarray(c['U0']), D)
    biaj = jnp.asarray(bia)
    tb = time.perf_counter()
    if mode == 'milp':
      z = jnp.asarray(milp_z)
    else:
      # tightened problem: every biactive pair closed and unloaded; its multipliers tell which side pays
      _, _, dLds, yg, _ = solve_j(qd, biaj, biaj, 0.0 * biaj)
      press_val = M_F * np.asarray(jnp.maximum(-dLds, 0.0) @ E2C)    # (T, nc)
      open_val = M_G * np.asarray(yg)
      z = jnp.asarray((bia > 0.5) & (press_val > open_val) & (press_val > 0)).astype(float)
      if mode == 'local':
        zs = np.asarray(z)
        idx = np.argwhere(bia > 0.5)
        cands = [zs]
        for (t_, j_) in idx:
          zc = zs.copy(); zc[t_, j_] = 1.0 - zc[t_, j_]; cands.append(zc)
        Z = jnp.asarray(np.stack(cands))
        _, objs, _, _, _ = solve_batch(qd, biaj, Z)
        objs = jnp.where(jnp.isfinite(objs), objs, jnp.inf)
        z = Z[int(jnp.argmin(objs))]
    t_branch = time.perf_counter() - tb
    tb = time.perf_counter()
    du, obj, _, _, info = solve_j(qd, biaj, z, z)
    du = np.asarray(du); t_qp = time.perf_counter() - tb
    U1 = np.clip(c['U0'] + du, ulb, uub)
    X1, _ = truem._rollout(jax.device_put(jnp.asarray(task.x_0), CPU), jax.device_put(jnp.asarray(U1), CPU), jnp.asarray(0.0))
    ok = bool(np.isfinite(float(obj)) and float(info['mu']) < 1e-8 and float(info['dyn']) < 1e-4 and float(info['prim']) < 1e-6)
    return dict(nbi=nbi, ok=ok, U1=U1, goal_true=goal(np.asarray(X1)), pressed=int(np.asarray(z).sum()), z=np.asarray(z),
                pred=float(obj), msg=f'{mode}: mu {float(info["mu"]):.1e} dyn {float(info["dyn"]):.1e} prim {float(info["prim"]):.1e}',
                time=time.perf_counter() - t0, t_lin=c['t_lin'], t_branch=t_branch, t_qp=t_qp)
  step.problem = problem
  return step
