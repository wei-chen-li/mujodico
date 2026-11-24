"""Minimal box-constrained iLQR / DDP (Gauss-Newton) on the smoothed planning objective -- the same rollout and cost
as ssqn.py, so DDP and L-BFGS can be compared head to head without PLF-V's model management.

  dynamics   x' = f_s(x, u): smoothed lifted contact law (LMS_KW planning model, eps = base * sig), forces eliminated
  cost       sum_k (x_k-g)'Q(x_k-g) + (u_k-u_ref)'R(u_k-u_ref) + (x_N-g)'Qf(x_N-g)     (quaternion-aware state cost)
  backward   Gauss-Newton Riccati (no dynamics curvature), Quu + lam I, per-stage box QP (projected Newton, lfddp_v._boxqp)
  forward    closed loop u = u + a k + K (x^ - x)  (default)  or open loop (--ol 1); backtracking a in ALPHAS, Armijo
  lam        /10 after an accepted step (floor 1e-9), x10 after a failed line search (stop above 1e10)
True-law read-out of the accepted plan every --ro-every iterations; incumbent = best true metric.

usage: python ilqr.py <Task> <a:b> <label> <out.jsonl> [--sigs 1e-3] [--iters 300] [--ol 0] [--ffm 0.01] [--start hold|npz]
       [--alphas 1,0.5,0.25,0.1,0.03,0.01] [--ro-every 5] [--stop 0.003] [--verbose 0] [--law lifted|full]
--law full: dynamics = trans.true_step(x, u, eps) (all constraint forces: joint limits, equalities -- the law
IppasLiftedMultipleShooting.rollout uses); lifted (default): contact forces only (what LFDDP plans with)."""
import os, sys, json, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, '..', '..', 'handoff_2026-11-22', 'scripts')
sys.path.insert(0, HERE); sys.path.insert(0, SC)
import numpy as np, jax, jax.numpy as jp
jax.config.update('jax_enable_x64', True)
import bench_tasks as T
from reloc_lms import LMS_KW
from simtools import sim_true, metric
from mujodico.examples.methods import IppasLiftedMultipleShooting, util


def opt(name, default, typ=float):
  a = sys.argv
  return typ(a[a.index(name) + 1]) if name in a else default


def boxqp(H, g, Hx, lo, hi, iters=4):
  """min 1/2 k'Hk + g'k, lo <= k <= hi (projected Newton, as lfddp_v._boxqp); returns k and free-set gains K."""
  n = g.shape[0]; I = jp.eye(n)
  k = jp.clip(-jp.linalg.solve(H, g), lo, hi)
  free = jp.ones(n, bool)
  for _ in range(iters):
    gr = g + H @ k
    clamped = ((k <= lo + 1e-12) & (gr > 0)) | ((k >= hi - 1e-12) & (gr < 0))
    free = ~clamped
    FF = free[:, None] & free[None, :]
    Hf = jp.where(FF, H, I)
    kc = jp.where(clamped, k, 0.0)
    rhs = jp.where(free, -(g + H @ kc), 0.0)
    k = jp.clip(jp.where(free, jp.linalg.solve(Hf, rhs), k), lo, hi)
  FF = free[:, None] & free[None, :]
  Hf = jp.where(FF, H, I)
  K = jp.where(free[:, None], -jp.linalg.solve(Hf, jp.where(free[:, None], Hx, 0.0)), 0.0)
  return k, K


class ILQR:
  def __init__(self, task, ffm=None, law='lifted'):
    self.t = t = task
    kw = dict(LMS_KW)
    if ffm is not None:
      kw['far_field_margin'] = ffm
    lms = IppasLiftedMultipleShooting(t, **kw); trans = lms.transcription
    self.base = trans.contact_smoothing
    N, nu, nx = t.horizon, t.nu, t.nx; ncf = trans.nw - nu
    self.N, self.nu, self.nx = N, nu, nx
    q = t.quaternion_w_idxs
    Q, Qf, R, g, ur = map(jp.asarray, (t.Q, t.Qf, t.R, t.x_goal, t.u_ref)); x0 = jp.asarray(t.x_0)
    lb, ub = (jp.asarray(b) for b in t.u_bounds)

    if law == 'full':   # smoothed law with ALL constraint forces (joint limits, equalities): trans.true_step
      ts = trans.true_step
      def f(x, u, eps):
        return ts(x, u, eps)[0]
    else:               # lifted contact law (LFDDP / the lifted IP): contact forces only
      def f(x, u, eps):
        cf = -trans.stage_equality(x, jp.concatenate([u, jp.zeros(ncf)]), None, eps)
        return trans.dynamics(x, jp.concatenate([u, cf]))
    lx = lambda x: util.state_cost(Q, g, x, q)
    lu = lambda u: util.control_cost(R, ur, u)
    lN = lambda x: util.state_cost(Qf, g, x, q)

    def cost(X, U):
      return jax.vmap(lx)(X[:-1]).sum() + jax.vmap(lu)(U).sum() + lN(X[-1])

    def roll(U, eps):
      def step(x, u):
        xn = f(x, u, eps); return xn, xn
      _, Xs = jax.lax.scan(step, x0, U)
      X = jp.concatenate([x0[None], Xs]); return X, cost(X, U)

    def fwd(X, U, k, K, alpha, eps, ol):
      def step(xh, inp):
        x, u, kk, KK = inp
        uh = jp.clip(u + alpha * kk + (1.0 - ol) * (KK @ (xh - x)), lb, ub)
        xn = f(xh, uh, eps)
        return xn, (xn, uh)
      _, (Xs, Us) = jax.lax.scan(step, x0, (X[:-1], U, k, K))
      Xn = jp.concatenate([x0[None], Xs]); return Xn, Us, cost(Xn, Us)

    def lin(X, U, eps):
      A, B = jax.vmap(jax.jacfwd(f, argnums=(0, 1)), (0, 0, None))(X[:-1], U, eps)
      lxv = jax.vmap(jax.grad(lx))(X[:-1]); lxx = jax.vmap(jax.hessian(lx))(X[:-1])
      luv = jax.vmap(jax.grad(lu))(U); luu = jax.vmap(jax.hessian(lu))(U)
      return A, B, lxv, lxx, luv, luu, jax.grad(lN)(X[-1]), jax.hessian(lN)(X[-1])

    def bwd(U, Lin, lam):
      A, B, lxv, lxx, luv, luu, vx, vxx = Lin
      def step(carry, inp):
        Vx, Vxx = carry
        a, b, lxk, lxxk, luk, luuk, u = inp
        Qx = lxk + a.T @ Vx; Qu = luk + b.T @ Vx
        Qxx = lxxk + a.T @ Vxx @ a; Quu = luuk + b.T @ Vxx @ b; Qux = b.T @ Vxx @ a
        H = 0.5 * (Quu + Quu.T) + lam * jp.eye(nu)
        k, K = boxqp(H, Qu, Qux, lb - u, ub - u)
        Vx_n = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx_n = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        d1 = k @ Qu; d2 = k @ Quu @ k
        return (Vx_n, 0.5 * (Vxx_n + Vxx_n.T)), (k, K, d1, d2)
      _, (k, K, d1, d2) = jax.lax.scan(step, (vx, vxx), (A, B, lxv, lxx, luv, luu, U), reverse=True)
      ok = jp.all(jp.isfinite(k)) & jp.all(jp.isfinite(K))
      return k, K, jp.sum(d1), jp.sum(d2), ok

    self.roll = jax.jit(roll); self.fwd = jax.jit(fwd); self.lin = jax.jit(lin); self.bwd = jax.jit(bwd)

  def true_metric(self, U):
    X = sim_true(self.t, U)
    return metric(self.t, X[-1]) if np.isfinite(X).all() else float('inf')

  def solve(self, U0, sigs, iters, alphas, ol=False, ro_every=5, stop=0.003, verbose=0):
    U = jp.asarray(U0, float); best = dict(r=self.true_metric(np.asarray(U)), U=np.asarray(U).copy(), at=('start', 0))
    log = []; tot = 0; tic = time.time(); stats = dict(n_acc=0, n_fail=0, alpha_hist={})
    for s in sigs:
      eps = jp.asarray(self.base * s); X, J = self.roll(U, eps); J = float(J); lam = 1e-6; it = 0; reason = 'iters'
      while it < iters:
        it += 1; tot += 1
        Lin = self.lin(X, U, eps)
        accepted = False
        while not accepted:
          k, K, d1, d2, ok = self.bwd(U, Lin, jp.asarray(lam))
          d1, d2 = float(d1), float(d2)
          if not bool(ok):
            lam *= 10; continue
          if abs(d1) < 1e-10 * max(abs(J), 1.0):
            break
          for a in alphas:
            Xn, Un, Jn = self.fwd(X, U, k, K, jp.asarray(a), eps, jp.asarray(float(ol)))
            Jn = float(Jn); exp = a * d1 + 0.5 * a * a * d2
            if verbose >= 2:
              print(f'    a {a:g} pred {exp:.3e} act {Jn - J:.3e}', flush=True)
            if np.isfinite(Jn) and Jn < J and Jn - J <= 1e-4 * exp:
              accepted = True; break
          if accepted:
            stats['n_acc'] += 1; stats['alpha_hist'][a] = stats['alpha_hist'].get(a, 0) + 1
            X, U, J = Xn, Un, Jn; lam = max(lam / 10.0, 1e-9)
          else:
            stats['n_fail'] += 1; lam *= 10
            if lam > 1e10:
              break
        if verbose:
          print(f'  sig {s:g} it {it} J {J:.4f} lam {lam:.1e} d1 {d1:.3e}', flush=True)
        if not accepted:
          reason = 'stalled (lam > 1e10 or d1 ~ 0)'; break
        if it % ro_every == 0:
          r = self.true_metric(np.asarray(U))
          if r < best['r']:
            best.update(r=r, U=np.asarray(U).copy(), at=(s, it))
          if best['r'] <= stop:
            reason = 'stop'; break
      r = self.true_metric(np.asarray(U))
      if r < best['r']:
        best.update(r=r, U=np.asarray(U).copy(), at=(s, it))
      log.append(dict(sig=s, its=it, J=J, true=r, best=best['r'], lam=lam, reason=reason))
      if best['r'] <= stop:
        break
    return best, log, tot, time.time() - tic, stats


def main():
  name = sys.argv[1]; a, b = (int(v) for v in sys.argv[2].split(':')); label = sys.argv[3]; out = sys.argv[4]
  sigs = [float(v) for v in opt('--sigs', '1e-3', str).split(',')]; iters = opt('--iters', 300, int)
  ol = bool(opt('--ol', 0, int)); ffm = opt('--ffm', None); start = opt('--start', 'hold', str)
  alphas = tuple(float(v) for v in opt('--alphas', '1,0.5,0.25,0.1,0.03,0.01', str).split(','))
  ro = opt('--ro-every', 5, int); stop = opt('--stop', 0.003); verbose = opt('--verbose', 0, int)
  law = opt('--law', 'lifted', str)
  save = opt('--save-dir', os.path.join(HERE, '..', 'results', 'plans'), str)
  for seed in range(a, b):
    t = T.TASKS[name](seed); S = ILQR(t, ffm=ffm, law=law)
    U0 = np.zeros((t.horizon, t.nu)) if start == 'hold' else np.load(start.format(seed=seed))['us']
    best, log, tot, dt, stats = S.solve(U0, sigs, iters, alphas, ol, ro, stop, verbose)
    row = dict(task=name, label=label, seed=seed, true_goal=best['r'], true_ok=bool(best['r'] < 0.01), sigs=sigs,
               ol=ol, ffm=ffm, law=law, its=tot, time=dt, best_at=best['at'], log=log,
               stats={k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()}) for k, v in stats.items()})
    print(json.dumps(row), flush=True)
    with open(out, 'a') as fh:
      fh.write(json.dumps(row) + '\n')
    os.makedirs(save, exist_ok=True)
    np.savez(os.path.join(save, f'{label}_{name}_{seed}.npz'), us=best['U'])


if __name__ == '__main__':
  main()
