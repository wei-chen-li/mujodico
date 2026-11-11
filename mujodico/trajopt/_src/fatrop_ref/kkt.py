"""Structured primal-dual KKT solves for the jitted IP-PAS solver.

This module solves FATROP's *augmented* system (after the slack and bound
multiplier rows have been eliminated, cf. ``PdSolverOrig::solve_once_impl``)

    [ H + Dx     J^T ] [ d  ]     [ f ]
    [   J       -De  ] [ dy ] = - [ g ]

for an optimal-control problem with stages k = 0..N-1 and terminal stage N:

* stage variables ``(dx_k, dw_k)``, terminal ``dx_N``;
* initial-state equality row      ``dx_0 - De0 * dlam0 + c0 = 0``;
* dynamics rows (always hard)     ``A_k dx_k + B_k dw_k - dx_{k+1} + a_k = 0``;
* path equality rows              ``E_k dx_k + F_k dw_k - De_k * dbeta_k + e_k = 0``;
* reduced inequality rows         ``Gx_k dx_k + Gw_k dw_k - Dii_k * dgamma_k + g_k = 0``.

The stationarity rows are

    (Q_k+diag(Dxx_k)) dx_k + S_k dw_k + A_k^T dlam_k + E_k^T dbeta_k
        + Gx_k^T dgamma_k - dlam_{k-1} [+ dlam0 if k == 0] + q_k = 0,
    S_k^T dx_k + (R_k+diag(Dxw_k)) dw_k + B_k^T dlam_k + F_k^T dbeta_k
        + Gw_k^T dgamma_k + r_k = 0,
    (Qf+diag(Dxf)) dx_N - dlam_{N-1} + qf = 0.

``De`` is zero in the original problem.  In FATROP's restoration phase the
elastic ``p/n`` variables are eliminated into ``De`` (``PdSolverResto``), so
the very same structured solver handles both phases.  The inequality block is
always folded into the stage Hessian through the positive diagonal ``Dii``.

Two interchangeable backends are provided: a serial generalized Riccati
recursion (``lax.scan``) and the parallel associative scan (PAS) whose
sequential depth is O(log N).  Both return the same FATROP-style inertia
flag: every stage's reduced control Hessian (projected onto the null space of
the hard path equalities, or penalized by ``De^{-1}`` when ``De > 0``) must
pass a pivoted Cholesky test (``check_reg`` with ``pivot_tol``), and, when the
initial row is regularized, so must the stage-0 value Hessian.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class KKTSystem(NamedTuple):
  Q: jax.Array      # (N, nx, nx)
  S: jax.Array      # (N, nx, nw)
  R: jax.Array      # (N, nw, nw)
  Qf: jax.Array     # (nx, nx)
  Dxx: jax.Array    # (N, nx)   diagonal primal regularization on x_k
  Dxw: jax.Array    # (N, nw)   diagonal primal regularization on w_k
  Dxf: jax.Array    # (nx,)     diagonal primal regularization on x_N
  A: jax.Array      # (N, nx, nx)
  B: jax.Array      # (N, nx, nw)
  E: jax.Array      # (N, ne, nx)
  F: jax.Array      # (N, ne, nw)
  Gx: jax.Array     # (N, ni, nx)
  Gw: jax.Array     # (N, ni, nw)
  De: jax.Array     # (N, ne)
  Dii: jax.Array    # (N, ni)  > 0
  De0: jax.Array    # (nx,)
  resto: jax.Array  # bool scalar: De > 0 on every path/initial row


class KKTRhs(NamedTuple):
  q: jax.Array      # (N, nx)
  r: jax.Array      # (N, nw)
  qf: jax.Array     # (nx,)
  a: jax.Array      # (N, nx)
  e: jax.Array      # (N, ne)
  g: jax.Array      # (N, ni)
  c0: jax.Array     # (nx,)


class KKTSolution(NamedTuple):
  dx: jax.Array     # (N+1, nx)
  dw: jax.Array     # (N, nw)
  dlam: jax.Array   # (N, nx)
  dbeta: jax.Array  # (N, ne)
  dgamma: jax.Array # (N, ni)
  dlam0: jax.Array  # (nx,)


def _sym(a):
  return 0.5 * (a + a.T)


def _check_reg(M, pivot_tol):
  """FATROP ``check_reg``: Cholesky pivots must be positive and not tiny."""
  M = _sym(M)
  if M.shape[0] == 0:
    return jnp.array(True)
  L = jnp.linalg.cholesky(M)
  piv = jnp.diag(L)
  ok = jnp.all(jnp.isfinite(L)) & jnp.all(piv > 0.0)
  return ok & jnp.all(piv * piv > pivot_tol * jnp.diag(M))


def _reduced_pd_ok(Huu, F, De, resto, pivot_tol):
  """Inertia test of one stage's reduced control Hessian.

  ``Huu`` already contains the downstream value Hessian, the primal
  regularization, and the folded inequality rows.  With hard path
  equalities (original problem) the test is on the null-space projection;
  with regularized rows (restoration) the rows are folded as a penalty,
  exactly as FATROP's ``D_e`` branch of ``AugSystemSolver::solve``.
  """
  nw = Huu.shape[0]
  ne = F.shape[0]
  if ne == 0:
    return _check_reg(Huu, pivot_tol)

  def hard(_):
    q, _r = jnp.linalg.qr(F.T, mode='complete')
    Z = q[:, ne:]
    if Z.shape[1] == 0:
      return jnp.array(True)
    return _check_reg(Z.T @ Huu @ Z, pivot_tol)

  def soft(_):
    Dinv = 1.0 / jnp.maximum(De, jnp.finfo(De.dtype).tiny)
    return _check_reg(Huu + F.T @ (Dinv[:, None] * F), pivot_tol)

  return jax.lax.cond(resto, soft, hard, None)


def _local_factor(Q, S, R, Dxx, Dxw, q, r, A, B, a, E, F, e, Gx, Gw, g, De,
                  Dii):
  """Eliminate ``(dw_k, dbeta_k, dgamma_k)`` given ``dx_k`` and ``dlam_k``.

  Returns the stage element ``(P, p, C, Ac, bc)`` of the PAS recursion

      dlam_{k-1} = P dx_k + p + Ac^T dlam_k     (value/costate relation)
      dx_{k+1}   = Ac dx_k + bc - C dlam_k,

  and the recovery maps ``(Zx, z0, Zl)`` with
  ``[dw; dbeta; dgamma] = -(Zx dx_k + z0 + Zl dlam_k)``.
  """
  nx = Q.shape[0]
  nw = R.shape[0]
  ne = F.shape[0]
  ni = Gw.shape[0]
  Qr = Q + jnp.diag(Dxx)
  Rr = R + jnp.diag(Dxw)
  Sr = S
  qr = q
  rr = r
  if ni:
    D = 1.0 / Dii
    DGx = D[:, None] * Gx
    DGw = D[:, None] * Gw
    Dg = D * g
    Qr = Qr + Gx.T @ DGx
    Sr = Sr + Gx.T @ DGw
    Rr = Rr + Gw.T @ DGw
    qr = qr + Gx.T @ Dg
    rr = rr + Gw.T @ Dg
  M = jnp.block([[Rr, F.T], [F, -jnp.diag(De)]])
  Kx = jnp.concatenate([Sr.T, E], axis=0)
  k0 = jnp.concatenate([rr, e], axis=0)
  Kl = jnp.concatenate([B.T, jnp.zeros((ne, nx), dtype=B.dtype)], axis=0)
  Zeq = jnp.linalg.solve(M, jnp.concatenate([Kx, k0[:, None], Kl], axis=1))
  Zeq_x = Zeq[:, :nx]
  zeq_0 = Zeq[:, nx]
  Zeq_l = Zeq[:, nx + 1:]
  # Stationarity of x_k after eliminating (dw, dbeta) with the inequality
  # rows folded in:  dlam_{k-1} = Qr dx + qr + A^T dlam - Kx^T Zeq(...).
  P = Qr - Kx.T @ Zeq_x
  p = qr - Kx.T @ zeq_0
  Bbar = jnp.concatenate([B, jnp.zeros((nx, ne), dtype=B.dtype)], axis=1)
  C = Bbar @ Zeq_l
  Ac = A - Bbar @ Zeq_x
  bc = a - Bbar @ zeq_0
  # Full recovery maps including the folded inequality multipliers:
  #   dgamma = D (Gx dx + Gw dw + g).
  if ni:
    Zw = Zeq[:nw]
    Zgx = D[:, None] * (Gw @ Zw[:, :nx] - Gx)
    zg0 = D * (Gw @ Zw[:, nx] - g)
    Zgl = D[:, None] * (Gw @ Zw[:, nx + 1:])
    Zx = jnp.concatenate([Zeq_x, Zgx], axis=0)
    z0 = jnp.concatenate([zeq_0, zg0], axis=0)
    Zl = jnp.concatenate([Zeq_l, Zgl], axis=0)
  else:
    Zx, z0, Zl = Zeq_x, zeq_0, Zeq_l
  # Note: the P/p above use the *folded* Hessian, which is the correct
  # Schur complement (the inequality multipliers were eliminated exactly).
  return _sym(P), p, _sym(C), Ac, bc, Zx, z0, Zl


def _compose(e2, e1):
  P1, p1, C1, A1, b1 = e1
  P2, p2, C2, A2, b2 = e2
  n = P1.shape[0]
  L = jnp.eye(n, dtype=P1.dtype) + P2 @ C1
  XL = jnp.linalg.solve(L, jnp.concatenate(
      [P2 @ A1, (p2 + P2 @ b1)[:, None], A2.T], axis=1))
  Xpa = XL[:, :n]
  Xp = XL[:, n]
  Xat = XL[:, n + 1:]
  P = P1 + A1.T @ Xpa
  p = p1 + A1.T @ Xp
  C = C2 + A2 @ (C1 @ Xat)
  Ao = A2 @ (A1 - C1 @ Xpa)
  bo = b2 + A2 @ (b1 - C1 @ Xp)
  return _sym(P), p, _sym(C), Ao, bo


def _initial_step(V0, v0, c0, De0):
  """Solve the initial row ``dx0 - De0 dlam0 + c0 = 0`` with
  ``dlam0 = -(V0 dx0 + v0)``."""
  nx = V0.shape[0]
  M = jnp.eye(nx, dtype=V0.dtype) + De0[:, None] * V0
  dx0 = jnp.linalg.solve(M, -(c0 + De0 * v0))
  dlam0 = -(V0 @ dx0 + v0)
  return dx0, dlam0


def _stage_inertia(sys: KKTSystem, Vnext, pivot_tol):
  def one(R, Dxw, B, F, Gw, Dii, De, Vn):
    Huu = R + jnp.diag(Dxw) + B.T @ Vn @ B
    if Gw.shape[0]:
      Huu = Huu + Gw.T @ ((1.0 / Dii)[:, None] * Gw)
    return _reduced_pd_ok(Huu, F, De, sys.resto, pivot_tol)
  oks = jax.vmap(one)(sys.R, sys.Dxw, sys.B, sys.F, sys.Gw, sys.Dii, sys.De,
                      Vnext)
  return jnp.all(oks)


def _initial_inertia(sys: KKTSystem, V0, pivot_tol):
  def soft(_):
    Dinv = 1.0 / jnp.maximum(sys.De0, jnp.finfo(sys.De0.dtype).tiny)
    return _check_reg(V0 + jnp.diag(Dinv), pivot_tol)
  return jax.lax.cond(sys.resto, soft, lambda _: jnp.array(True), None)


class PASFactor(NamedTuple):
  """Right-hand-side independent part of the PAS solve."""
  lu_M: jax.Array    # (N, nw+ne, nw+ne) LU of the stage saddle block
  piv_M: jax.Array
  Zx: jax.Array      # (N, nw+ne+ni, nx) recovery maps (matrix parts)
  Zl: jax.Array      # (N, nw+ne+ni, nx)
  Kx: jax.Array      # (N, nw+ne, nx) = [Sr^T; E]
  V: jax.Array       # (N+1, nx, nx) value Hessians
  K: jax.Array       # (N, nx, nx) closed-loop transitions (I + C V+)^-1 Ac
  lu_L: jax.Array    # (N, nx, nx) LU of I + C_k V_{k+1}
  piv_L: jax.Array
  C: jax.Array       # (N, nx, nx)
  lu_0: jax.Array    # LU of I + De0 V0
  piv_0: jax.Array
  ok: jax.Array


def _stage_matrices(Q, S, R, Dxx, Dxw, A, B, E, F, Gx, Gw, De, Dii):
  nx = Q.shape[0]
  nw = R.shape[0]
  ne = F.shape[0]
  ni = Gw.shape[0]
  Qr = Q + jnp.diag(Dxx)
  Rr = R + jnp.diag(Dxw)
  Sr = S
  if ni:
    D = 1.0 / Dii
    Qr = Qr + Gx.T @ (D[:, None] * Gx)
    Sr = Sr + Gx.T @ (D[:, None] * Gw)
    Rr = Rr + Gw.T @ (D[:, None] * Gw)
  M = jnp.block([[Rr, F.T], [F, -jnp.diag(De)]])
  Kx = jnp.concatenate([Sr.T, E], axis=0)
  Kl = jnp.concatenate([B.T, jnp.zeros((ne, nx), dtype=B.dtype)], axis=0)
  lu, piv = jax.scipy.linalg.lu_factor(M)
  Z = jax.scipy.linalg.lu_solve((lu, piv), jnp.concatenate([Kx, Kl], axis=1))
  Zeq_x, Zeq_l = Z[:, :nx], Z[:, nx:]
  P = Qr - Kx.T @ Zeq_x
  Bbar = jnp.concatenate([B, jnp.zeros((nx, ne), dtype=B.dtype)], axis=1)
  C = Bbar @ Zeq_l
  Ac = A - Bbar @ Zeq_x
  if ni:
    Zw = Z[:nw]
    Zx = jnp.concatenate([Zeq_x, D[:, None] * (Gw @ Zw[:, :nx] - Gx)], 0)
    Zl = jnp.concatenate([Zeq_l, D[:, None] * (Gw @ Zw[:, nx:])], 0)
  else:
    Zx, Zl = Zeq_x, Zeq_l
  return lu, piv, _sym(P), _sym(C), Ac, Zx, Zl, Kx


def _compose_mat(e2, e1):
  P1, C1, A1 = e1
  P2, C2, A2 = e2
  n = P1.shape[0]
  L = jnp.eye(n, dtype=P1.dtype) + P2 @ C1
  XL = jnp.linalg.solve(L, jnp.concatenate([P2 @ A1, A2.T], axis=1))
  Xpa, Xat = XL[:, :n], XL[:, n:]
  P = P1 + A1.T @ Xpa
  C = C2 + A2 @ (C1 @ Xat)
  Ao = A2 @ (A1 - C1 @ Xpa)
  return _sym(P), _sym(C), Ao


def factor_pas(sys: KKTSystem, pivot_tol: float = 1e-8) -> PASFactor:
  N, nx, _ = sys.Q.shape
  lu_M, piv_M, P, C, Ac, Zx, Zl, Kx = jax.vmap(_stage_matrices)(
      sys.Q, sys.S, sys.R, sys.Dxx, sys.Dxw, sys.A, sys.B, sys.E, sys.F,
      sys.Gx, sys.Gw, sys.De, sys.Dii)
  Pf = _sym(sys.Qf + jnp.diag(sys.Dxf))
  zmat = jnp.zeros((1, nx, nx), dtype=P.dtype)
  V, _, _ = jax.lax.associative_scan(
      jax.vmap(_compose_mat), (jnp.concatenate([P, Pf[None]], 0),
                               jnp.concatenate([C, zmat], 0),
                               jnp.concatenate([Ac, zmat], 0)), reverse=True)
  I = jnp.eye(nx, dtype=P.dtype)
  # (I + C_k V_{k+1}) is the transpose of (I + V_{k+1} C_k), so one LU gives
  # both the closed-loop transition K and the costate recursion matrix K^T.
  lu_L, piv_L = jax.vmap(jax.scipy.linalg.lu_factor)(
      I[None] + jnp.einsum('nij,njk->nik', C, V[1:]))
  K = jax.vmap(lambda lu, pv, a: jax.scipy.linalg.lu_solve((lu, pv), a))(
      lu_L, piv_L, Ac)
  lu_0, piv_0 = jax.scipy.linalg.lu_factor(I + sys.De0[:, None] * V[0])
  ok = _stage_inertia(sys, V[1:], pivot_tol) & _initial_inertia(
      sys, V[0], pivot_tol)
  return PASFactor(lu_M, piv_M, Zx, Zl, Kx, V, K, lu_L, piv_L, C, lu_0, piv_0,
                   ok)


def solve_factored(fac: PASFactor, sys: KKTSystem, rhs: KKTRhs) -> KKTSolution:
  N, nx, _ = sys.Q.shape
  nw = sys.R.shape[-1]
  ne = sys.F.shape[1]
  ni = sys.Gw.shape[1]

  def stage_vec(lu, piv, Kx, q, r, a, e, g, Gx, Gw, B, Dii):
    qr, rr = q, r
    if ni:
      Dg = g / Dii
      qr = qr + Gx.T @ Dg
      rr = rr + Gw.T @ Dg
    z0 = jax.scipy.linalg.lu_solve((lu, piv), jnp.concatenate([rr, e]))
    p = qr - Kx.T @ z0
    bc = a - B @ z0[:nw]
    if ni:
      z0 = jnp.concatenate([z0, (Gw @ z0[:nw] - g) / Dii])
    return p, bc, z0

  p, bc, z0 = jax.vmap(stage_vec)(fac.lu_M, fac.piv_M, fac.Kx, rhs.q, rhs.r,
                                  rhs.a, rhs.e, rhs.g, sys.Gx, sys.Gw, sys.B,
                                  sys.Dii)
  V = fac.V
  KT = jnp.swapaxes(fac.K, 1, 2)
  # Costate recursion v_k = K_k^T (v_{k+1} + V_{k+1} bc_k) + p_k, v_N = qf,
  # as a reverse scan over affine maps (matrix products only).
  m = p + jnp.einsum('nij,nj->ni', KT, jnp.einsum('nij,nj->ni', V[1:], bc))
  Ms = jnp.concatenate([KT, jnp.zeros((1, nx, nx), KT.dtype)], 0)
  ms = jnp.concatenate([m, rhs.qf[None]], 0)

  def rcomp(a, b):
    # a: later stage, b: earlier stage; returns b o a.
    Ma, ma = a
    Mb, mb = b
    return Mb @ Ma, jnp.einsum('...ij,...j->...i', Mb, ma) + mb

  _, v = jax.lax.associative_scan(rcomp, (Ms, ms), reverse=True)
  dx0 = jax.scipy.linalg.lu_solve((fac.lu_0, fac.piv_0),
                                  -(rhs.c0 + sys.De0 * v[0]))
  kk = jax.vmap(lambda lu, pv, b_, C, vn: jax.scipy.linalg.lu_solve(
      (lu, pv), b_ - C @ vn))(fac.lu_L, fac.piv_L, bc, fac.C, v[1:])
  ks = jnp.concatenate([dx0[None], kk], 0)
  Ks = jnp.concatenate([jnp.eye(nx, dtype=V.dtype)[None], fac.K], 0)

  def acomp(a, b):
    k1, K1 = a
    k2, K2 = b
    return k2 + jnp.einsum('...ij,...j->...i', K2, k1), K2 @ K1

  dx, _ = jax.lax.associative_scan(acomp, (ks, Ks))
  dlam = jnp.einsum('nij,nj->ni', V[1:], dx[1:]) + v[1:]
  dlam0 = -(V[0] @ dx0 + v[0])
  zz = -(jnp.einsum('nij,nj->ni', fac.Zx, dx[:-1]) + z0 +
         jnp.einsum('nij,nj->ni', fac.Zl, dlam))
  return KKTSolution(dx, zz[:, :nw], dlam, zz[:, nw:nw + ne], zz[:, nw + ne:],
                     dlam0)


def solve_pas(sys: KKTSystem, rhs: KKTRhs, pivot_tol: float = 1e-8):
  fac = factor_pas(sys, pivot_tol)
  return solve_factored(fac, sys, rhs), fac.ok


def solve_serial(sys: KKTSystem, rhs: KKTRhs, pivot_tol: float = 1e-8):
  """Backward generalized Riccati recursion (reference backend)."""
  N, nx, _ = sys.Q.shape
  nw = sys.R.shape[-1]
  ne = sys.F.shape[1]
  Pf = _sym(sys.Qf + jnp.diag(sys.Dxf))

  def backward(carry, inp):
    Vn, vn = carry
    (Q, S, R, Dxx, Dxw, q, r, A, B, a, E, F, e, Gx, Gw, g, De, Dii) = inp
    P, p, C, Ac, bc, Zx, z0, Zl = _local_factor(
        Q, S, R, Dxx, Dxw, q, r, A, B, a, E, F, e, Gx, Gw, g, De, Dii)
    L = jnp.eye(nx, dtype=P.dtype) + Vn @ C
    X = jnp.linalg.solve(L, jnp.concatenate(
        [Vn @ Ac, (vn + Vn @ bc)[:, None]], axis=1))
    V = _sym(P + Ac.T @ X[:, :nx])
    v = p + Ac.T @ X[:, nx]
    return (V, v), (V, v)

  inputs = (sys.Q, sys.S, sys.R, sys.Dxx, sys.Dxw, rhs.q, rhs.r, sys.A, sys.B,
            rhs.a, sys.E, sys.F, rhs.e, sys.Gx, sys.Gw, rhs.g, sys.De,
            sys.Dii)
  _, (Vs, vs) = jax.lax.scan(backward, (Pf, rhs.qf), inputs, reverse=True)
  Vall = jnp.concatenate([Vs, Pf[None]], 0)
  vall = jnp.concatenate([vs, rhs.qf[None]], 0)
  dx0, dlam0 = _initial_step(Vall[0], vall[0], rhs.c0, sys.De0)

  def forward(xk, inp):
    (Q, S, R, Dxx, Dxw, q, r, A, B, a, E, F, e, Gx, Gw, g, De, Dii, Vn,
     vn) = inp
    P, p, C, Ac, bc, Zx, z0, Zl = _local_factor(
        Q, S, R, Dxx, Dxw, q, r, A, B, a, E, F, e, Gx, Gw, g, De, Dii)
    L = jnp.eye(nx, dtype=P.dtype) + C @ Vn
    xn = jnp.linalg.solve(L, Ac @ xk + bc - C @ vn)
    ln = Vn @ xn + vn
    zz = -(Zx @ xk + z0 + Zl @ ln)
    return xn, (xn, ln, zz)

  _, (xt, dlam, zz) = jax.lax.scan(forward, dx0, inputs + (Vall[1:],
                                                          vall[1:]))
  dx = jnp.concatenate([dx0[None], xt], 0)
  sol = KKTSolution(dx, zz[:, :nw], dlam, zz[:, nw:nw + ne], zz[:, nw + ne:],
                    dlam0)
  ok = _stage_inertia(sys, Vall[1:], pivot_tol) & _initial_inertia(
      sys, Vall[0], pivot_tol)
  return sol, ok


# ---------------------------------------------------------------------------
# Exact condensation of equality rows with an identity block
# ---------------------------------------------------------------------------
class Condensation(NamedTuple):
  """Change of variables ``dw = Tw du + Tx dx + t0`` eliminating ``dw[f]``.

  Valid when the path equalities read ``E dx + Fu dw[u] + dw[f] + e = 0``
  (an identity block on the ``f`` columns, one row per ``f`` column) and are
  hard (``De = 0``), as for lifted contact forces ``cf - f_contact(x,u) = 0``.
  """
  sys: KKTSystem     # reduced system with ne = 0 and nw = len(u)
  Tw: jax.Array      # (N, nw, nu_r)
  Tx: jax.Array      # (N, nw, nx)
  Rr: jax.Array      # (N, nw, nw) original R + diag(Dxw)


def condense(sys: KKTSystem, iu, if_) -> Condensation:
  N, nx, _ = sys.Q.shape
  nw = sys.R.shape[-1]
  nur = len(iu)
  iu = jnp.asarray(iu)
  if_ = jnp.asarray(if_)
  Fu = sys.F[:, :, iu]                                    # (N, ne, nur)
  Tw = jnp.zeros((N, nw, nur), sys.R.dtype)
  Tw = Tw.at[:, iu, :].set(jnp.eye(nur, dtype=sys.R.dtype)[None])
  Tw = Tw.at[:, if_, :].set(-Fu)
  Tx = jnp.zeros((N, nw, nx), sys.R.dtype).at[:, if_, :].set(-sys.E)
  Rr = sys.R + jax.vmap(jnp.diag)(sys.Dxw)
  tT = lambda M: jnp.swapaxes(M, 1, 2)
  RTx = Rr @ Tx
  Q = sys.Q + sys.S @ Tx + tT(sys.S @ Tx) + tT(Tx) @ RTx
  S = sys.S @ Tw + tT(Tx) @ (Rr @ Tw)
  R = tT(Tw) @ Rr @ Tw
  red = KKTSystem(
      Q=0.5 * (Q + tT(Q)), S=S, R=0.5 * (R + tT(R)), Qf=sys.Qf, Dxx=sys.Dxx,
      Dxw=jnp.zeros((N, nur), sys.R.dtype), Dxf=sys.Dxf,
      A=sys.A + sys.B @ Tx, B=sys.B @ Tw,
      E=jnp.zeros((N, 0, nx), sys.E.dtype), F=jnp.zeros((N, 0, nur), sys.F.dtype),
      Gx=sys.Gx + sys.Gw @ Tx, Gw=sys.Gw @ Tw,
      De=jnp.zeros((N, 0), sys.De.dtype), Dii=sys.Dii, De0=sys.De0,
      resto=sys.resto)
  return Condensation(red, Tw, Tx, Rr)


def condense_rhs(cd: Condensation, sys: KKTSystem, rhs: KKTRhs, if_) -> KKTRhs:
  N, nw = rhs.r.shape
  t0 = jnp.zeros((N, nw), rhs.r.dtype).at[:, jnp.asarray(if_)].set(-rhs.e)
  rt = rhs.r + jnp.einsum('nij,nj->ni', cd.Rr, t0)
  tTv = lambda M, v: jnp.einsum('nji,nj->ni', M, v)
  q = rhs.q + jnp.einsum('nij,nj->ni', sys.S, t0) + tTv(cd.Tx, rt)
  return KKTRhs(q=q, r=tTv(cd.Tw, rt), qf=rhs.qf,
                a=rhs.a + jnp.einsum('nij,nj->ni', sys.B, t0),
                e=jnp.zeros((N, 0), rhs.e.dtype),
                g=rhs.g + jnp.einsum('nij,nj->ni', sys.Gw, t0), c0=rhs.c0)


def expand(cd: Condensation, sys: KKTSystem, rhs: KKTRhs, sol: KKTSolution,
           if_) -> KKTSolution:
  N, nw = rhs.r.shape
  if_ = jnp.asarray(if_)
  t0 = jnp.zeros((N, nw), rhs.r.dtype).at[:, if_].set(-rhs.e)
  mv = lambda M, v: jnp.einsum('nij,nj->ni', M, v)
  mtv = lambda M, v: jnp.einsum('nji,nj->ni', M, v)
  dw = mv(cd.Tw, sol.dw) + mv(cd.Tx, sol.dx[:-1]) + t0
  # Equality multipliers from the (eliminated) identity-column stationarity.
  row = (mtv(sys.S, sol.dx[:-1]) + mv(cd.Rr, dw) + mtv(sys.B, sol.dlam) +
         mtv(sys.Gw, sol.dgamma) + rhs.r)
  dbeta = -row[:, if_]
  return KKTSolution(sol.dx, dw, sol.dlam, dbeta, sol.dgamma, sol.dlam0)
