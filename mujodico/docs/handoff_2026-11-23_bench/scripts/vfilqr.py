"""Minimal iLQR (as ilqr.py) + planning-only VIRTUAL FORCES for contact discovery at a distance.

Why (HANDOFF_bench.md 5.7): from a far start the smoothed objective is flat at physical smoothing (|g|/J ~ 1e-4) and,
at coarse smoothing, its gradient points AWAY from the object: smoothed contact forces are push-only normals (+
friction), so a distant pad can only shove the object off its goal.  A lift / pull has no far-field representative.

Mechanism (generic, nothing task-specific):
  objects   every body with a free joint that has at least one contact pair with a non-free body ("robot")
  controls  u_aug = [u, v], v_o in R^3 a world-frame force at object o's free-joint origin (bounded |v| <= vmax_o)
  dynamics  x' = f_s(x, u) (smoothed lifted law, as ilqr.py), then a semi-implicit kick of the object:
            qvel_lin += dt v / m_o,  qpos_lin += dt^2 v / m_o   (planning only)
  penalty   sum_o w (dmin_o(x)_+ + delta) |v_o|^2, dmin_o = soft-min over the robot-object pairs' signed distances
            (MJX reports true distances for every pair in the contact list), Gauss-Newton Hessian of the residual
            r_o = sqrt(w (dmin_o + delta)) v_o  (gives the state-control cross term that pulls the robot in)
  schedule  w in --ws (e.g. 1,10,100,1000) at sigma --vsig, then v := 0 and the plain sigma ladder --sigs
True-law read-outs (incumbent) always use the real controls u only (v dropped).
--mode adhesion: instead of a free force, controls s >= 0 per robot-object contact pair pull the object toward that
robot geom along the pair normal with magnitude s exp(-d / ell); applied as qfrc_applied inside a smoothed full MJX
step; penalty w s^2.  RESULT (HANDOFF_bench.md 5.7): both modes negative on GripperCubeLiftLong seed 2.

usage: python vfilqr.py <Task> <a:b> <label> <out.jsonl> [--ws 1,10,100,1000] [--vsig 1] [--sigs 1,0.1,0.01,1e-3]
       [--delta 0.005] [--tau 0.005] [--vmax 3] (x m_o g) [--dagg min|mean] [--mode free|adhesion] [--ell 0.05] [--verbose 1|2] [--phase-dir DIR] [--iters 150] [--ro-every 5] [--stop 0.003] [--law lifted|full]"""
import os, sys, json, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, '..', '..', 'handoff_2026-11-22', 'scripts')
sys.path.insert(0, HERE); sys.path.insert(0, SC)
import numpy as np, jax, jax.numpy as jp, mujoco
jax.config.update('jax_enable_x64', True)
import bench_tasks as T
from reloc_lms import LMS_KW
from simtools import sim_true, metric
from ilqr import boxqp, opt
from mujodico import mjx
from mujodico.examples.methods import IppasLiftedMultipleShooting, util


def virtual_objects(task):
  """[(body, qpos_adr, dof_adr, mass, [contact-pair indices with robot geoms])] for free bodies touched by the robot."""
  m = task.model
  free_body = {int(m.jnt_bodyid[j]): j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
  mm = mjx.put_model(m, impl='jax', contact_approximation=task.contact_approx, contact_smoothing=task.contact_smoothing)
  d = mjx.make_data(mm, impl='jax'); d = mjx.forward(mm, d.replace(qpos=jp.asarray(task.x_0[:m.nq])))
  c = d._impl.contact
  g1, g2 = np.asarray(c.geom1), np.asarray(c.geom2)
  out = []
  for b, j in free_body.items():
    pairs = []
    for i in range(g1.size):
      b1, b2 = int(m.geom_bodyid[g1[i]]), int(m.geom_bodyid[g2[i]])
      if b in (b1, b2):
        other = b2 if b1 == b else b1
        if other != 0 and other not in free_body:      # robot geom (not world, not another object)
          pairs.append(i)
    if pairs:
      out.append((b, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), float(m.body_subtreemass[b]), pairs))
  return mm, out


class VFILQR:
  def __init__(self, task, ffm=None, law='lifted', delta=0.005, tau=0.005, vmax=3.0, dagg='min', mode='free', ell=0.05):
    self.t = t = task
    kw = dict(LMS_KW)
    if ffm is not None:
      kw['far_field_margin'] = ffm
    lms = IppasLiftedMultipleShooting(t, **kw); trans = lms.transcription
    self.base = trans.contact_smoothing
    m = t.model; N, nu, nx = t.horizon, t.nu, t.nx; ncf = trans.nw - nu; nq = m.nq; dt = m.opt.timestep
    mm, objs = virtual_objects(t); self.objs = objs; no = len(objs)
    npair = sum(len(o[4]) for o in objs)
    nv3 = 3 * no if mode == 'free' else npair; nua = nu + nv3
    self.N, self.nu, self.nua = N, nu, nua
    q = t.quaternion_w_idxs
    Q, Qf, R, g, ur = map(jp.asarray, (t.Q, t.Qf, t.R, t.x_goal, t.u_ref)); x0 = jp.asarray(t.x_0)
    lb0, ub0 = (np.asarray(b, float) for b in t.u_bounds)
    if mode == 'free':
      vm = np.concatenate([np.full(3, vmax * o[3] * 9.81) for o in objs]) if no else np.zeros(0); vlo = -vm
    else:      # adhesion strengths s >= 0, one per robot-object contact pair
      vm = np.concatenate([np.full(len(o[4]), vmax * o[3] * 9.81) for o in objs]) if no else np.zeros(0); vlo = 0 * vm
    self.lb_v = jp.asarray(np.concatenate([lb0, vlo])); self.ub_v = jp.asarray(np.concatenate([ub0, vm]))
    self.lb_0 = jp.asarray(np.concatenate([lb0, np.zeros(nv3)])); self.ub_0 = jp.asarray(np.concatenate([ub0, np.zeros(nv3)]))
    qa = np.array([o[1] for o in objs], int); da = np.array([o[2] for o in objs], int); ms = np.array([o[3] for o in objs])
    pair_idx = [np.array(o[4], int) for o in objs]
    g1 = np.asarray(mjx.forward(mm, mjx.make_data(mm, impl='jax'))._impl.contact.geom1)
    g2 = np.asarray(mjx.forward(mm, mjx.make_data(mm, impl='jax'))._impl.contact.geom2)
    robot_groups = []                      # per object: pair positions grouped by robot geom
    for o, p in zip(objs, pair_idx):
      rgeom = [int(g2[i]) if int(m.geom_bodyid[g1[i]]) == o[0] else int(g1[i]) for i in p]
      robot_groups.append([np.array([j for j, r in enumerate(rgeom) if r == u_], int) for u_ in sorted(set(rgeom))])
    dmake = mjx.make_data(mm, impl='jax')

    if law == 'full':
      ts = trans.true_step
      def f0(x, u, eps):
        return ts(x, u, eps)[0]
    else:
      def f0(x, u, eps):
        cf = -trans.stage_equality(x, jp.concatenate([u, jp.zeros(ncf)]), None, eps)
        return trans.dynamics(x, jp.concatenate([u, cf]))

    obj_is_g2 = [np.array([int(m.geom_bodyid[g2[i]]) == o[0] for i in p]) for o, p in zip(objs, pair_idx)]

    def obj_acc(x, ua):
      if mode == 'free':
        return [ua[nu + 3 * i: nu + 3 * i + 3] / ms[i] for i in range(no)]
      dd = mjx.fwd_position(mm, dmake.replace(qpos=x[:nq], qvel=x[nq:nq + m.nv]))
      c = dd._impl.contact; out = []; j = nu
      for i in range(no):
        p = pair_idx[i]; n = c.frame[p, 0, :]                      # normal geom1 -> geom2
        toward_robot = jp.where(jp.asarray(obj_is_g2[i])[:, None], -n, n)    # object -> robot geom
        env = jp.exp(-jp.maximum(c.dist[p], 0.0) / ell)
        sp = ua[j:j + p.size]; j += p.size
        out.append(jp.sum((sp * env)[:, None] * toward_robot, 0) / ms[i])
      return out

    ffm_v = float(LMS_KW['far_field_margin'] if ffm is None else ffm)

    def f_adh(x, ua, eps):      # smoothed full MJX step with the adhesion pulls as applied forces (contacts respond)
      accs = obj_acc(x, ua); qf = jp.zeros(m.nv)
      for i in range(no):
        qf = qf.at[da[i]:da[i] + 3].set(accs[i] * ms[i])
      me = mm.replace(opt=mm.opt.replace(_impl=mm.opt._impl.replace(contact_smoothing=eps, contact_far_field_margin=ffm_v)))
      dd = mjx.step(me, dmake.replace(qpos=x[:nq], qvel=x[nq:nq + m.nv], ctrl=ua[:nu], qfrc_applied=qf))
      return jp.concatenate([dd.qpos, dd.qvel, dd.act])

    def f(x, ua, eps):
      if mode != 'free':
        return f_adh(x, ua, eps)
      xn = f0(x, ua[:nu], eps)
      accs = obj_acc(x, ua)
      for i in range(no):
        acc = accs[i]
        xn = xn.at[nq + da[i]: nq + da[i] + 3].add(dt * acc)
        xn = xn.at[qa[i]: qa[i] + 3].add(dt * dt * acc)
      return xn

    def dmin(x):
      dd = mjx.fwd_position(mm, dmake.replace(qpos=x[:nq], qvel=x[nq:nq + m.nv]))
      dist = dd._impl.contact.dist
      if not no:
        return jp.zeros(0)
      if dagg == 'mean':     # mean over the robot geoms of each geom's (soft-min) distance to the object
        out = []
        for p, rg in zip(pair_idx, robot_groups):
          out.append(jp.mean(jp.stack([-tau * jax.scipy.special.logsumexp(-dist[p[gi]] / tau) for gi in rg])))
        return jp.stack(out)
      return jp.stack([-tau * jax.scipy.special.logsumexp(-dist[p] / tau) for p in pair_idx])

    def vres(x, ua, w):      # virtual-force residual, cost = |r|^2
      if not no:
        return jp.zeros(0)
      if mode != 'free':
        return jp.sqrt(w) * ua[nu:]
      s = jp.sqrt(w) * jp.sqrt(jp.maximum(dmin(x), 0.0) + delta)     # finite derivative at w = 0
      return (ua[nu:].reshape(no, 3) * s[:, None]).ravel()

    lx = lambda x: util.state_cost(Q, g, x, q)
    lu = lambda ua: util.control_cost(R, ur, ua[:nu])
    lN = lambda x: util.state_cost(Qf, g, x, q)

    def cost(X, U, w):
      r = jax.vmap(vres, (0, 0, None))(X[:-1], U, w)
      return jax.vmap(lx)(X[:-1]).sum() + jax.vmap(lu)(U).sum() + jp.sum(r * r) + lN(X[-1])

    def roll(U, eps, w):
      def step(x, ua):
        xn = f(x, ua, eps); return xn, xn
      _, Xs = jax.lax.scan(step, x0, U)
      X = jp.concatenate([x0[None], Xs]); return X, cost(X, U, w)

    def fwd(X, U, k, K, alpha, eps, w, lb, ub):
      def step(xh, inp):
        x, u, kk, KK = inp
        uh = jp.clip(u + alpha * kk + KK @ (xh - x), lb, ub)
        xn = f(xh, uh, eps)
        return xn, (xn, uh)
      _, (Xs, Us) = jax.lax.scan(step, x0, (X[:-1], U, k, K))
      Xn = jp.concatenate([x0[None], Xs]); return Xn, Us, cost(Xn, Us, w)

    def lin(X, U, eps, w):
      A, B = jax.vmap(jax.jacfwd(f, argnums=(0, 1)), (0, 0, None))(X[:-1], U, eps)
      r = jax.vmap(vres, (0, 0, None))(X[:-1], U, w)
      Jx, Ju = jax.vmap(jax.jacfwd(vres, argnums=(0, 1)), (0, 0, None))(X[:-1], U, w)
      lxv = jax.vmap(jax.grad(lx))(X[:-1]) + 2 * jp.einsum('kri,kr->ki', Jx, r)
      lxx = jax.vmap(jax.hessian(lx))(X[:-1]) + 2 * jp.einsum('kri,krj->kij', Jx, Jx)
      luv = jax.vmap(jax.grad(lu))(U) + 2 * jp.einsum('kri,kr->ki', Ju, r)
      luu = jax.vmap(jax.hessian(lu))(U) + 2 * jp.einsum('kri,krj->kij', Ju, Ju)
      lux = 2 * jp.einsum('kri,krj->kij', Ju, Jx)
      return A, B, lxv, lxx, luv, luu, lux, jax.grad(lN)(X[-1]), jax.hessian(lN)(X[-1])

    def bwd(U, Lin, lam, lb, ub):
      A, B, lxv, lxx, luv, luu, lux, vx, vxx = Lin
      def step(carry, inp):
        Vx, Vxx = carry
        a, b, lxk, lxxk, luk, luuk, luxk, u = inp
        Qx = lxk + a.T @ Vx; Qu = luk + b.T @ Vx
        Qxx = lxxk + a.T @ Vxx @ a; Quu = luuk + b.T @ Vxx @ b; Qux = luxk + b.T @ Vxx @ a
        H = 0.5 * (Quu + Quu.T) + lam * jp.eye(nua)
        k, K = boxqp(H, Qu, Qux, lb - u, ub - u)
        Vx_n = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx_n = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        return (Vx_n, 0.5 * (Vxx_n + Vxx_n.T)), (k, K, k @ Qu, k @ Quu @ k)
      _, (k, K, d1, d2) = jax.lax.scan(step, (vx, vxx), (A, B, lxv, lxx, luv, luu, lux, U), reverse=True)
      ok = jp.all(jp.isfinite(k)) & jp.all(jp.isfinite(K))
      return k, K, jp.sum(d1), jp.sum(d2), ok

    self.roll = jax.jit(roll); self.fwd = jax.jit(fwd); self.lin = jax.jit(lin); self.bwd = jax.jit(bwd)
    self.dmin = jax.jit(jax.vmap(dmin)); self.phase_dir = None

  def true_metric(self, U):
    X = sim_true(self.t, np.asarray(U)[:, :self.nu])
    return metric(self.t, X[-1]) if np.isfinite(X).all() else float('inf')

  def solve(self, U0, ws, vsig, sigs, iters, alphas=(1, 0.5, 0.25, 0.1, 0.03, 0.01), ro_every=5, stop=0.003, verbose=1):
    U = jp.asarray(U0, float); best = dict(r=self.true_metric(U), U=np.asarray(U).copy(), at=('start', 0))
    log = []; tot = 0; tic = time.time()
    phases = [(vsig, w, self.lb_v, self.ub_v) for w in ws] + [(s, 0.0, self.lb_0, self.ub_0) for s in sigs]
    for s, w, lb, ub in phases:
      U = jp.clip(U, lb, ub)
      eps = jp.asarray(self.base * s); wj = jp.asarray(w)
      X, J = self.roll(U, eps, wj); J = float(J); lam = 1e-6; it = 0; reason = 'iters'
      while it < iters:
        it += 1; tot += 1
        Lin = self.lin(X, U, eps, wj); accepted = False
        while not accepted:
          k, K, d1, d2, ok = self.bwd(U, Lin, jp.asarray(lam), lb, ub)
          d1, d2 = float(d1), float(d2)
          if not bool(ok):
            lam *= 10; continue
          if abs(d1) < 1e-10 * max(abs(J), 1.0):
            break
          for a in alphas:
            Xn, Un, Jn = self.fwd(X, U, k, K, jp.asarray(a), eps, wj, lb, ub)
            Jn = float(Jn)
            if verbose >= 2:
              print(f'      it {it} lam {lam:.1e} a {a:g} d1 {d1:.3e} pred {a * d1 + 0.5 * a * a * d2:.3e} act {Jn - J:.3e}', flush=True)
            if np.isfinite(Jn) and Jn < J and Jn - J <= 1e-4 * (a * d1 + 0.5 * a * a * d2):
              accepted = True; break
          if accepted:
            X, U, J = Xn, Un, Jn; lam = max(lam / 10.0, 1e-9)
          else:
            lam *= 10
            if lam > 1e10:
              break
        if not accepted:
          reason = 'stalled'; break
        if it % ro_every == 0:
          r = self.true_metric(U)
          if r < best['r']:
            best.update(r=r, U=np.asarray(U).copy(), at=(s, w, it))
          if best['r'] <= stop:
            reason = 'stop'; break
      r = self.true_metric(U)
      if r < best['r']:
        best.update(r=r, U=np.asarray(U).copy(), at=(s, w, it))
      vn = float(jp.abs(U[:, self.nu:]).max()) if self.nua > self.nu else 0.0
      dm = np.asarray(self.dmin(X)).min(0).round(4).tolist()
      row = dict(sig=s, w=w, its=it, J=J, true=r, best=best['r'], vmax=vn, dmin=dm, reason=reason, t=round(time.time() - tic, 1))
      log.append(row)
      if verbose:
        print('  ', json.dumps(row), flush=True)
      if self.phase_dir:
        np.savez(os.path.join(self.phase_dir, f'phase_{len(log):02d}.npz'), U=np.asarray(U), X=np.asarray(X))
      if best['r'] <= stop:
        break
    return best, log, tot, time.time() - tic


def main():
  name = sys.argv[1]; a, b = (int(v) for v in sys.argv[2].split(':')); label = sys.argv[3]; out = sys.argv[4]
  ws = [float(v) for v in opt('--ws', '1,10,100,1000', str).split(',')]
  sigs = [float(v) for v in opt('--sigs', '1,0.1,0.01,1e-3', str).split(',')]
  vsig = opt('--vsig', 1.0); iters = opt('--iters', 150, int); ffm = opt('--ffm', None)
  delta = opt('--delta', 0.005); tau = opt('--tau', 0.005); vmax = opt('--vmax', 3.0)
  ro = opt('--ro-every', 5, int); stop = opt('--stop', 0.003); law = opt('--law', 'lifted', str)
  mode = opt('--mode', 'free', str); ell = opt('--ell', 0.05)
  dagg = opt('--dagg', 'min', str); verbose = opt('--verbose', 1, int); phase_dir = opt('--phase-dir', None, str)
  save = opt('--save-dir', os.path.join(HERE, '..', 'results', 'plans'), str)
  for seed in range(a, b):
    t = T.TASKS[name](seed); S = VFILQR(t, ffm=ffm, law=law, delta=delta, tau=tau, vmax=vmax, dagg=dagg, mode=mode, ell=ell)
    if phase_dir:
      S.phase_dir = phase_dir.format(seed=seed); os.makedirs(S.phase_dir, exist_ok=True)
    print(f'{name} seed {seed}: virtual objects', [(t.model.body(o[0]).name, len(o[4])) for o in S.objs], flush=True)
    U0 = np.zeros((t.horizon, S.nua))
    best, log, tot, dt = S.solve(U0, ws, vsig, sigs, iters, ro_every=ro, stop=stop, verbose=verbose)
    row = dict(task=name, label=label, seed=seed, true_goal=best['r'], true_ok=bool(best['r'] < 0.01), ws=ws, vsig=vsig,
               sigs=sigs, delta=delta, tau=tau, vmax=vmax, dagg=dagg, mode=mode, ell=ell, ffm=ffm, law=law, its=tot, time=dt, best_at=best['at'], log=log)
    print(json.dumps(row), flush=True)
    with open(out, 'a') as fh:
      fh.write(json.dumps(row) + '\n')
    os.makedirs(save, exist_ok=True)
    np.savez(os.path.join(save, f'{label}_{name}_{seed}.npz'), us=best['U'][:, :t.nu], v=best['U'][:, t.nu:])


if __name__ == '__main__':
  main()
