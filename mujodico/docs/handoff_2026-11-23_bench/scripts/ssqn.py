"""SSQN: single-shooting quasi-Newton (L-BFGS-B) on the smoothed planning objective, with a smoothing ladder and
true-law read-outs.

Objective per level s: J_s(U) = sum_k [ (x_k - g)' Q (x_k - g) + (u_k - u_ref)' R (u_k - u_ref) ] + (x_N - g)' Qf (x_N - g)
(+ optional discovery barrier -bmu sum log n), x = rollout of U under the smoothed lifted law (LMS_KW planning model,
eps = base * s, far-field margin ffm), box bounds on U.  Every `ro_every` iterations (and at the end of each level) the
current U is rolled out under the TRUE law; the incumbent is the best true metric seen (as PLF-V's read-outs).

Motivation (HANDOFF_bench.md 5.3-5.4): on the grasped cube lift PLF-V's DDP stalls at a "squirt" cliff while plain
L-BFGS from hold-still solves it at every smoothing level.

usage: python ssqn.py <Task> <a:b> <label> <out.jsonl> [--start hold|<npz pattern with {seed}>] [--sigs 1,0.1,0.01,1e-3]
       [--iters 150] [--ffm 0.01] [--bmu 0] [--ro-every 10] [--stop 0.003] [--save-dir DIR]
Runs on the CPU device (jax_platforms cpu)."""
import os, sys, json, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'handoff_2026-11-22', 'scripts'))
import numpy as np, jax, jax.numpy as jp
jax.config.update('jax_enable_x64', True)
from scipy.optimize import minimize
import bench_tasks as T
from reloc_lms import LMS_KW
from simtools import sim_true, metric
from mujodico.examples.methods import IppasLiftedMultipleShooting, util


def opt(name, default, typ=float):
  a = sys.argv
  return typ(a[a.index(name) + 1]) if name in a else default


class SSQN:
  def __init__(self, task, ffm=None, bmu=0.0):
    self.t = t = task
    kw = dict(LMS_KW)
    if ffm is not None:
      kw['far_field_margin'] = ffm
    self.lms = lms = IppasLiftedMultipleShooting(t, **kw)
    self.N, self.nu = N, nu = t.horizon, t.nu
    lms.rollout(t.x_0, np.zeros((N, nu)), 1.0)
    roll = lms._rollout; self.base = lms.transcription.contact_smoothing; q = t.quaternion_w_idxs
    Q, Qf, R, g, ur = map(jp.asarray, (t.Q, t.Qf, t.R, t.x_goal, t.u_ref)); x0 = jp.asarray(t.x_0)
    G = None
    if bmu > 0:
      from mujodico.trajopt._src import transcriptions as trx
      mm = trx.mjx.put_model(t.model, impl='jax', contact_approximation=t.contact_approx, contact_smoothing=t.contact_smoothing)
      _, cts = trx._extract_contact_indices(mm); ncf = lms.transcription.nw - nu
      G = np.zeros((len(cts), ncf))
      for i, c in enumerate(cts):
        if c.type == trx._Contact.Type.PYRAMIDAL:
          G[i, c.cf_index:c.cf_index + 2 * (c.dim - 1)] = 1.0
        else:
          G[i, c.cf_index] = 1.0
      G = jp.asarray(G)

    def J(uflat, eps):
      U = uflat.reshape(N, nu); X, C = roll(x0, U, eps)
      st = jax.vmap(lambda x: util.state_cost(Q, g, x, q))(X[:-1]).sum() + jax.vmap(lambda u: util.control_cost(R, ur, u))(U).sum()
      val = st + util.state_cost(Qf, g, X[-1], q)
      if G is not None:
        val = val - bmu * jp.sum(jp.log(jp.maximum(C @ G.T, 1e-30)))
      return jp.where(jp.isfinite(val), val, 1e12)
    self.vg = jax.jit(jax.value_and_grad(J))
    lb, ub = t.u_bounds
    self.bounds = list(zip(np.tile(lb, N), np.tile(ub, N)))

  def true_metric(self, U):
    X = sim_true(self.t, U)
    return metric(self.t, X[-1]) if np.isfinite(X).all() else float('inf')

  def solve(self, U0, sigs, iters, ro_every=10, stop=0.003):
    t = self.t; U = np.array(U0, float)
    best = dict(r=self.true_metric(U), U=U.copy(), at=('start', 0))
    log = []; n_it = 0; tic = time.time()

    class Done(Exception):
      pass

    for s in sigs:
      eps = jp.asarray(self.base * s); k = [0]

      def f(u):
        v, gr = self.vg(jp.asarray(u), eps)
        return float(v), np.asarray(gr, np.float64)

      def cb(u):
        k[0] += 1
        if k[0] % ro_every == 0:
          r = self.true_metric(u.reshape(self.N, self.nu))
          if r < best['r']:
            best.update(r=r, U=u.reshape(self.N, self.nu).copy(), at=(s, k[0]))
          if best['r'] <= stop:
            raise Done()
      try:
        res = minimize(f, U.ravel(), jac=True, method='L-BFGS-B', bounds=self.bounds, callback=cb, options=dict(maxiter=iters))
        U = res.x.reshape(self.N, self.nu); Jv = float(res.fun); msg = res.message[:24]
      except Done:
        msg = 'stop'; Jv = float('nan')
        U = best['U'].copy()
      r = self.true_metric(U)
      if r < best['r']:
        best.update(r=r, U=U.copy(), at=(s, k[0]))
      n_it += k[0]
      log.append(dict(sig=s, its=k[0], J=Jv, true=r, best=best['r'], msg=msg))
      if best['r'] <= stop:
        break
    return best, log, n_it, time.time() - tic


def main():
  name = sys.argv[1]; a, b = (int(v) for v in sys.argv[2].split(':')); label = sys.argv[3]; out = sys.argv[4]
  start = opt('--start', 'hold', str); sigs = [float(v) for v in opt('--sigs', '1,0.1,0.01,1e-3', str).split(',')]
  iters = opt('--iters', 150, int); ffm = opt('--ffm', None); bmu = opt('--bmu', 0.0); ro = opt('--ro-every', 10, int)
  stop = opt('--stop', 0.003); save = opt('--save-dir', os.path.join(HERE, '..', 'results', 'plans'), str)
  for seed in range(a, b):
    t = T.TASKS[name](seed)
    S = SSQN(t, ffm=ffm, bmu=bmu)   # the jitted objective closes over x_0 / goal: rebuilt per seed
    if start == 'hold':
      U0 = np.zeros((t.horizon, t.nu))
    else:
      U0 = np.load(start.format(seed=seed))['us']
    r0 = S.true_metric(U0)
    best, log, n_it, dt = S.solve(U0, sigs, iters, ro, stop)
    row = dict(task=name, label=label, seed=seed, true_goal=best['r'], true_ok=bool(best['r'] < 0.01), start=start,
               start_goal=r0, sigs=sigs, iters=iters, ffm=ffm, bmu=bmu, its=n_it, time=dt, best_at=best['at'], log=log)
    print(json.dumps(row), flush=True)
    with open(out, 'a') as fh:
      fh.write(json.dumps(row) + '\n')
    os.makedirs(save, exist_ok=True)
    np.savez(os.path.join(save, f'{label}_{name}_{seed}.npz'), us=best['U'])


if __name__ == '__main__':
  main()
