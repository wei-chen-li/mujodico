"""LPEC branch step at a plan, as a function (the in-process version of lpec.py --quad 1, gurobipy).

make_lpec(task, S, truem, lms) -> step(us, D, log=None) -> dict(nbi, U1, goal_true, pressed, pred, time, msg) or
  dict(nbi=0) when the plan has no biactive contact (B-stationary: no branch to take).

Linearization about the plan's TRUE-law trajectory (X*, U*, CF* = true-law contact forces); per stage k, contact j:
  non-biactive: dcf = Phi_x dx + Phi_u du (contact law linearized at smoothing 1e-3 of the base)
  biactive (gap < GAP_TOL, normal force < F_TOL), binary z: press (gap stays closed, extra lifted force s >= 0 on its
  edges) or separate (s = 0, gap stays >= 0).
Objective: second-order model of the task cost (PSD-projected stage/terminal blocks); trust region |du|_inf <= D within
the control bounds.  Branch set from the MILP (linear objective), then the convex QP with those binaries fixed.
The step's controls U* + du are returned with their true-law goal; the caller re-solves from them."""
import os, time
import numpy as np
import scipy.sparse as sp
import jax, jax.numpy as jp
from mujodico.examples.methods import util
from mujodico.trajopt._src import transcriptions as trx

GAP_TOL, F_TOL, M_F, M_G = 1e-3, 1e-6, 5.0, 0.05


def _grb(c, A, lo, hi, lb, ub, integ, Qobj=None, time_limit=None, extra=None):
  import gurobipy as gp
  from gurobipy import GRB
  m = gp.Model(env=_env()); m.setParam('TimeLimit', time_limit or int(os.environ.get('GRB_TL', 300)))
  m.setParam('MIPGap', 1e-4); m.setParam('DualReductions', 0)
  vt = np.where(integ > 0, GRB.BINARY, GRB.CONTINUOUS)
  x = m.addMVar(len(c), lb=np.where(np.isinf(lb), -GRB.INFINITY, lb), ub=np.where(np.isinf(ub), GRB.INFINITY, ub), vtype=vt)
  A = sp.csr_matrix(A, copy=True); A.data[np.abs(A.data) < 1e-9] = 0.0; A.eliminate_zeros()
  eq = np.flatnonzero(lo == hi); le = np.flatnonzero(np.isinf(lo) & ~np.isinf(hi)); ge = np.flatnonzero(~np.isinf(lo) & (lo != hi))
  rng = np.flatnonzero(~np.isinf(lo) & ~np.isinf(hi) & (lo != hi))
  if eq.size: m.addConstr(A[eq] @ x == lo[eq])
  if le.size: m.addConstr(A[le] @ x <= hi[le])
  if ge.size: m.addConstr(A[ge] @ x >= lo[ge])
  if rng.size: m.addConstr(A[rng] @ x <= hi[rng])
  obj = c @ x
  if Qobj is not None:
    Qc = sp.csr_matrix(Qobj, copy=True); Qc.data[np.abs(Qc.data) < 1e-8] = 0.0; Qc.eliminate_zeros()
    obj = obj + 0.5 * (x @ Qc @ x)
    m.setParam('Method', 2); m.setParam('NumericFocus', 1)
  m.setObjective(obj, GRB.MINIMIZE)
  for k_, v_ in (extra or {}).items():
    m.setParam(k_, v_)
  m.optimize()
  ok = m.SolCount > 0
  return (np.asarray(x.X) if ok else None), f'status {m.Status} obj {m.ObjVal if ok else float("nan"):.4g}'


_ENV = None
def _env():
  global _ENV
  if _ENV is None:
    import gurobipy as gp
    _ENV = gp.Env(empty=True); _ENV.setParam('OutputFlag', 0); _ENV.start()
  return _ENV


def make_lpec(task, S, truem, lms):
  trans = lms.transcription
  nu, ncf, N, nx, nq = S.nu, S.ncf, S.N, task.nx, task.model.nq
  mm = trx.mjx.put_model(task.model, impl='jax', contact_approximation=task.contact_approx, contact_smoothing=task.contact_smoothing)
  cf_addr, contacts = trx._extract_contact_indices(mm)
  d0 = trx.mjx.make_data(mm, impl=mm.impl)
  first = np.array([c.cf_index for c in contacts])
  edges = [list(range(c.cf_index, c.cf_index + (2 * (c.dim - 1) if c.type == trx._Contact.Type.PYRAMIDAL else 1))) for c in contacts]
  gapf = lambda x: trx.mjx.fwd_position(mm, d0.replace(qpos=x[:nq]))._impl.efc_pos[cf_addr][first]
  ts = truem.transcription.true_step
  eps = jp.asarray(S.base * 1e-3)
  CPU = jax.devices('cpu')[0]

  @jax.jit
  def linearize(X, U, CF, P):
    def st(x, u, c, xn):
      w = jp.concatenate([u, c])
      fv = trans.dynamics(x, w)
      fx, fw = jax.jacfwd(trans.dynamics, argnums=(0, 1))(x, w)
      Lx, Lu = jax.jacfwd(S.law, argnums=(0, 1))(x, u, eps)
      g, gx = gapf(x), jax.jacfwd(gapf)(x)
      lx, lu = jax.grad(S.lst, (0, 1))(x, u, P)
      (hxx, _), (_, huu) = jax.hessian(S.lst, (0, 1))(x, u, P)
      return fv - xn, fx, fw[:, :nu], fw[:, nu:], Lx, Lu, g, gx, lx, lu, hxx, huu
    out = jax.vmap(st)(X[:-1], U, CF, X[1:])
    return out + (jax.grad(S.lterm)(X[-1], P), jax.hessian(S.lterm)(X[-1], P))
  true_cf = jax.jit(jax.vmap(lambda x, u: ts(x, u, jp.asarray(0.0))[1]))
  Qm = np.asarray(getattr(task, 'metric_Q', task.Q))
  goal = lambda Xe: float(util.state_cost(Qm, task.x_goal, np.asarray(Xe[-1]), task.quaternion_w_idxs))
  ulb, uub = (np.asarray(b_, float) for b_ in task.u_bounds)
  cache = {}

  def build(us):
    """linearization + constraint matrix at the plan us (cached: shrinking the radius reuses it)."""
    key = np.asarray(us).tobytes()
    if key in cache:
      return cache[key]
    cache.clear()
    us = jax.device_put(jp.asarray(np.asarray(us)), CPU)
    X, _ = truem._rollout(jax.device_put(jp.asarray(task.x_0), CPU), us, jp.asarray(0.0))
    CF = true_cf(X[:-1], us)
    P = trans.params
    dfc, fx, fu, fc, Lx, Lu, gap, gx, lx, lu, hxx, huu, lN, hNN = [np.asarray(a) for a in linearize(X, us, CF, P)]
    nrm = np.asarray(CF) @ np.asarray(S.G).T
    bi = list(zip(*np.nonzero((gap < GAP_TOL) & (nrm < F_TOL))))
    ix = lambda k: k * nx
    iu0 = (N + 1) * nx; iu = lambda k: iu0 + k * nu
    ic0 = iu0 + N * nu; ic = lambda k: ic0 + k * ncf
    is0 = ic0 + N * ncf
    bedges = [(b, e) for b, (k, j) in enumerate(bi) for e in edges[j]]
    iz0 = is0 + len(bedges); nvar = iz0 + len(bi)
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = [0]
    def add(entries, l, h):
      for c_, v_ in entries:
        rows.append(r[0]); cols.append(c_); vals.append(v_)
      lo.append(l); hi.append(h); r[0] += 1
    for k in range(N):                                   # dynamics
      for i in range(nx):
        ent = [(ix(k + 1) + i, -1.0)]
        ent += [(ix(k) + a, fx[k, i, a]) for a in np.nonzero(fx[k, i])[0]]
        ent += [(iu(k) + a, fu[k, i, a]) for a in np.nonzero(fu[k, i])[0]]
        ent += [(ic(k) + a, fc[k, i, a]) for a in np.nonzero(fc[k, i])[0]]
        add(ent, -dfc[k, i], -dfc[k, i])
    bslack = {(bi[b][0], e): is0 + n_ for n_, (b, e) in enumerate(bedges)}
    for k in range(N):                                   # contact law (slack s on biactive edges)
      for e in range(ncf):
        ent = [(ic(k) + e, 1.0)]
        ent += [(ix(k) + a, -Lx[k, e, a]) for a in np.nonzero(Lx[k, e])[0]]
        ent += [(iu(k) + a, -Lu[k, e, a]) for a in np.nonzero(Lu[k, e])[0]]
        if (k, e) in bslack:
          ent.append((bslack[(k, e)], -1.0))
        add(ent, 0.0, 0.0)
    for b, (k, j) in enumerate(bi):                      # biactive branches
      z = iz0 + b
      for e in edges[j]:
        add([(bslack[(k, e)], 1.0), (z, -M_F)], -np.inf, 0.0)
      gent = [(ix(k) + a, gx[k, j, a]) for a in np.nonzero(gx[k, j])[0]]
      add(gent, -max(gap[k, j], 0.0), np.inf)
      add(gent + [(z, M_G)], -np.inf, M_G - gap[k, j])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r[0], nvar))
    c = np.zeros(nvar)
    for k in range(N):
      c[ix(k):ix(k) + nx] += lx[k]; c[iu(k):iu(k) + nu] += lu[k]
    c[ix(N):ix(N) + nx] += lN
    qr, qc, qv = [], [], []
    def addblk(H_, off):
      w_, V_ = np.linalg.eigh(0.5 * (H_ + H_.T))
      H_ = (V_ * np.maximum(w_, 0.0)) @ V_.T
      H_[np.abs(H_) < 1e-12] = 0.0
      ii, jj = np.nonzero(H_)
      qr.extend(off + ii); qc.extend(off + jj); qv.extend(H_[ii, jj])
    for k in range(N):
      addblk(hxx[k] + 1e-8 * np.eye(nx), ix(k)); addblk(huu[k] + 1e-8 * np.eye(nu), iu(k))
    addblk(hNN + 1e-8 * np.eye(nx), ix(N))
    Qobj = sp.coo_matrix((qv, (qr, qc)), shape=(nvar, nvar))
    out = dict(bi=bi, A=A, lo=np.array(lo), hi=np.array(hi), c=c, Q=Qobj, nvar=nvar, iu0=iu0, ic0=ic0, iz0=iz0,
               U0=np.asarray(us), iu=iu)
    cache[key] = out
    return out

  def step(us, D):
    t0 = time.perf_counter()
    m = build(us)
    bi = m['bi']
    if not bi:
      return dict(nbi=0, time=time.perf_counter() - t0)
    nvar, iu0, ic0, iz0, iu, U0 = m['nvar'], m['iu0'], m['ic0'], m['iz0'], m['iu'], m['U0']
    lb = np.full(nvar, -np.inf); ub = np.full(nvar, np.inf)
    lb[:iu0] = -0.5; ub[:iu0] = 0.5
    lb[0:nx] = 0.0; ub[0:nx] = 0.0
    for k in range(N):
      lb[iu(k):iu(k) + nu] = np.maximum(-D, ulb - U0[k]); ub[iu(k):iu(k) + nu] = np.minimum(D, uub - U0[k])
    is0 = ic0 + N * ncf
    lb[ic0:is0] = -10 * M_F; ub[ic0:is0] = 10 * M_F
    lb[is0:iz0] = 0.0; ub[is0:iz0] = M_F
    lb[iz0:] = 0.0; ub[iz0:] = 1.0
    integ = np.zeros(nvar); integ[iz0:] = 1
    x0, msg0 = _grb(m['c'], m['A'], m['lo'], m['hi'], lb, ub, integ)
    if x0 is None:
      return dict(nbi=len(bi), U1=None, msg='MILP: ' + msg0, time=time.perf_counter() - t0)
    lbq, ubq = lb.copy(), ub.copy(); zf = np.round(x0[iz0:]); lbq[iz0:] = zf; ubq[iz0:] = zf
    v, msg = _grb(m['c'], m['A'], m['lo'], m['hi'], lbq, ubq, np.zeros(nvar), m['Q'])
    if v is None:
      v, msg = _grb(m['c'], m['A'], m['lo'], m['hi'], lbq, ubq, np.zeros(nvar), m['Q'], extra=dict(NumericFocus=3, BarHomogeneous=1))
      if v is None:
        v, msg = x0, 'QP failed; MILP step'
    du = v[iu0:ic0].reshape(N, nu)
    U1 = np.clip(U0 + du, ulb, uub)
    X1, _ = truem._rollout(jax.device_put(jp.asarray(task.x_0), CPU), jax.device_put(jp.asarray(U1), CPU), jp.asarray(0.0))
    zs = np.round(v[iz0:]).astype(int)
    return dict(nbi=len(bi), U1=U1, goal_true=goal(np.asarray(X1)), pressed=int(zs.sum()), msg=msg,
                time=time.perf_counter() - t0)
  return step
