"""E4 / component C1: true-law acceptance on top of one-start LMS (HANDOFF section 4.C).

Discovery: phase D of the one-start recipe (100 iterations at smoothing 1, mu0 given).
Then bursts of k IP-PAS iterations at a smoothing level chosen by accept/reject:
  - each burst starts from the rollout of the incumbent's controls under the law at the burst's
    smoothing (an exactly feasible LMS point), barrier restarted at mu_burst;
  - candidate = true-law (unsmoothed, full-law) rollout of the burst's controls;
  - accept if its merit is lower (merit: true-law goal error, or true-law objective):
    next burst one level lower in smoothing; else reject, go back up one level, halve k.
The incumbent is always an exact true-law trajectory.  Stops at --stop-tol or when the budget
(--budget iterations after discovery) is spent or k < kmin.
usage: c1_accept.py <Task> <a:b> <label> <out.jsonl> [--mu0 1e-3] [--k 20] [--budget 250]
       [--ladder 1,0.1,0.01,0.001] [--merit goal|obj] [--no-adapt] [--disc-iters 100] [--from-plans DIR LABEL]
       [--mu-burst 1e-3]"""
import os, sys, json, time
import numpy as np
import jax, jax.numpy as jp
sys.path.insert(0, '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-22/scripts')
sys.path.insert(0, '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-14/followup/scripts')
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting, util
from mujodico.trajopt._src.fatrop_ref import ippas
import hard_tasks  # noqa
import extra_tasks  # noqa
from reloc_lms import LMS_KW

name, rng_, label, out = sys.argv[1:5]
args = sys.argv[5:]
def opt(k, d, f=str):
  return f(args[args.index(k) + 1]) if k in args else d
mu0 = opt('--mu0', 1e-3, float)
k0 = opt('--k', 20, int)
budget = opt('--budget', 250, int)
ladder = [float(v) for v in opt('--ladder', '1,0.1,0.01,0.001').split(',')]
merit_kind = opt('--merit', 'goal')
adapt = '--no-adapt' not in args
disc_iters = opt('--disc-iters', 100, int)
mu_burst = opt('--mu-burst', 1e-3, float)
stop_tol = opt('--stop-tol', 1e-3, float)
kmin = opt('--kmin', 5, int)
plans = (args[args.index('--from-plans') + 1], args[args.index('--from-plans') + 2]) if '--from-plans' in args else None
accept_all = '--accept-all' in args   # ablation: no true-law judgment, every burst is kept (fixed ladder)
disc_sched = json.loads(opt('--disc', 'null'))   # optional discovery phases (wrapper schedule), e.g. V0.1's 100/10 smoothing
qf_boost = opt('--qf-boost', 1.0, float)   # terminal weight multiplier for bursts once the incumbent is near (< --boost-below)
boost_below = opt('--boost-below', 0.03, float)
project = opt('--project', 'open')   # open: candidate = open-loop true-law rollout of the burst's controls;
                                     # lqr: also TVLQR-track the burst's plan under the full true law and use the
                                     # APPLIED controls (an exact true-law trajectory: open-loop replay reproduces it)
start = opt('--start', 'smooth')     # burst start: smooth = smoothed-law rollout of the incumbent controls;
                                     # true = the incumbent's true-law states with law forces at the burst smoothing
disc_mode = opt('--disc-mode', 'wrapper')   # wrapper: the wrapper's return (with best_tol it is often the static start!);
                                             # last: the discovery phase's last iterate; none: no discovery (bursts from x0 held, zero controls)
disc_mumin = opt('--disc-mumin', 1e-4, float)   # barrier floor of bursts before the first accepted burst (1e-2 = phase D's)
descend = '--descend-on-stall' in args   # nothing accepted and k at its cap: step one smoothing level DOWN, k back to k0
form = opt('--form', 'LMS')     # LMS | B2 (explicit consistency, explicit_lms.py)
up_ok = opt('--up', 1, int)     # allow going above the starting level on reject (up to max(ladder) * 10**(up-1))
hess_after = [float(v) for v in opt('--hess-after', '1,0,0').split(',')]   # B1: dynamics curvature weights after the first accept
roll_cpu = opt('--roll-cpu', 0, int)
post_ok = opt('--post-ok', 10 ** 9, int)
disc_frac = opt('--disc-frac', 0.0, float)   # first accept (end of discovery) needs this relative true-goal decrease (while k < 40)   # A5: at most this many bursts after the first one that ends below 0.01   # true rollouts, retractions, objective and TVLQR on the CPU device
import contextlib
RC = (lambda: jax.default_device(jax.devices('cpu')[0])) if roll_cpu else contextlib.nullcontext
a, b = (int(v) for v in rng_.split(':'))
J = lambda o: o.item() if hasattr(o, 'item') else (o.tolist() if hasattr(o, 'tolist') else str(o))

PH = lambda sm, mi, mm: dict(smoothing=sm, iters=100, hess=[1, 0, 0], mu_init=mi, mu_min=mm)
lms = None
for seed in range(a, b):
  task = make_task(name, seed)
  if lms is None:
    if form == 'B2':
      from explicit_lms import ExplicitLMS
      E = ExplicitLMS(task, variant='B2', max_iter=100)
      lms = E.lms
      solve_ph = lambda ph, X_, W_, k_: E.solver.solve(E.prm(ph['smoothing'], ph['mu_init'], ph['mu_min'], ph['hess']), X_, W_, max_iter=k_)
    else:
      lms = IppasLiftedMultipleShooting(task, smoothing_schedule=[PH(1, mu0, 1e-2)], max_iter=100, **LMS_KW)
      solve_ph = lambda ph, X_, W_, k_: lms.solver.solve(lms._params(ph), X_, W_, max_iter=k_)
    trans = lms.transcription
    nu = task.nu; N = task.horizon; ncf = trans.nw - nu
    truem = IppasLiftedMultipleShooting(task)
    def _law_roll(x0, us, eps):
      def step(x, u):
        cf = -trans.stage_equality(x, jp.concatenate([u, jp.zeros(ncf)]), None, eps)
        xn = trans.dynamics(x, jp.concatenate([u, cf]))
        return xn, (x, cf)
      xN, (X, cfs) = jax.lax.scan(step, x0, us)
      return jp.concatenate([X, xN[None]]), cfs
    law_roll = jax.jit(_law_roll)
    def _g_plan(x, u, eps):
      cf = -trans.stage_equality(x, jp.concatenate([u, jp.zeros(ncf)]), None, eps)
      return trans.dynamics(x, jp.concatenate([u, cf]))
    jac_fn = jax.jit(jax.vmap(jax.jacfwd(_g_plan, argnums=(0, 1)), in_axes=(0, 0, None)))
    law_forces = jax.jit(jax.vmap(lambda x, u, eps: -trans.stage_equality(x, jp.concatenate([u, jp.zeros(ncf)]), None, eps), in_axes=(0, 0, None)))
    _true_step = truem.transcription.true_step
    ulb, uub = (np.asarray(b_, float) for b_ in task.u_bounds) if task.u_bounds is not None else (np.full(nu, -np.inf), np.full(nu, np.inf))
    @jax.jit
    def closed_loop(xs_, us_, K_):
      def body(x, k):
        u = jp.clip(us_[k] + K_[k] @ (x - xs_[k]), ulb, uub)
        xn, _ = _true_step(x, u, jp.asarray(0.0))
        return xn, (xn, u)
      _, (X_, U_) = jax.lax.scan(body, xs_[0], jp.arange(us_.shape[0]))
      return jp.concatenate([xs_[:1], X_]), U_
    def lqr_project(xs_, us_, sm_):
      with RC():
        return _lqr_project(xs_, us_, sm_)
    def _lqr_project(xs_, us_, sm_):
      """TVLQR (planning-law linearization at sm_) tracking under the full true law; returns the
      true-law states and the applied controls."""
      A_, B_ = (np.asarray(a_) for a_ in jac_fn(jp.asarray(xs_[:-1]), jp.asarray(us_), jp.asarray(trans.contact_smoothing * sm_)))
      n_ = xs_.shape[1]; nq_ = task.model.nq
      Qd = np.diag(np.r_[np.ones(nq_), 0.1 * np.ones(n_ - nq_)]); Rr = 1e-2 * np.eye(nu)
      P = 10 * Qd; K_ = np.zeros((len(us_), nu, n_))
      for kk in range(len(us_) - 1, -1, -1):
        Sm = Rr + B_[kk].T @ P @ B_[kk]
        K_[kk] = -np.linalg.solve(Sm, B_[kk].T @ P @ A_[kk])
        P = Qd + A_[kk].T @ P @ (A_[kk] + B_[kk] @ K_[kk]); P = 0.5 * (P + P.T)
      X_, U_ = closed_loop(jp.asarray(xs_), jp.asarray(us_), jp.asarray(K_))
      return np.asarray(X_), np.asarray(U_)
  else:
    lms.update(task); truem.update(task)
    if form == 'B2': E.update(task)
  Qm = getattr(task, 'metric_Q', task.Q)
  sc = lambda x: float(util.state_cost(Qm, task.x_goal, x, task.quaternion_w_idxs)) if np.isfinite(x).all() else np.inf

  TM = dict(solve=0.0, true=0.0, retr=0.0, lqr=0.0)
  def true_eval(us):
    with RC():
      return _true_eval(us)
  def _true_eval(us):
    t0_ = time.perf_counter()
    X, cfs = truem.rollout(task.x_0, us, 0.0)
    if not np.isfinite(X).all():
      TM['true'] += time.perf_counter() - t0_
      return np.inf, np.inf, X
    g = sc(X[-1])
    obj = lms.objective(X, np.concatenate([us, cfs], 1))
    TM['true'] += time.perf_counter() - t0_
    return g, obj, X

  tic = time.perf_counter()
  its = 0
  if plans is not None:
    zz = np.load(f'{plans[0]}/{plans[1]}_{name}_{seed}.npz'); us = zz['us']; xs_plan = zz['xs'] if 'xs' in zz else None
  elif disc_mode == 'none':
    us = np.zeros((N, nu)); xs_plan = np.tile(np.asarray(task.x_0), (N + 1, 1))
  else:
    if form == 'B2':
      S0 = solve_ph(PH(1, mu0, 1e-2), np.tile(np.asarray(task.x_0), (N + 1, 1)), np.zeros((N, trans.nw)), disc_iters)
      us = np.array(S0.cur.w)[:, :nu]; its += int(S0.it); xs_plan = np.array(S0.cur.x)
    else:
      lms.smoothing_schedule = [PH(1, mu0, 1e-2)] if disc_sched is None else [dict(PH(1, mu0, 1e-2), **q) for q in disc_sched]
      if disc_mode == 'last':
        lms.smoothing_schedule = [dict(q, track_best=False) for q in lms.smoothing_schedule]
      r = lms.solve({'xs': np.tile(np.asarray(task.x_0), (N + 1, 1)), 'us': np.zeros((N, nu))}, iterations=disc_iters, verbose=False)
      us = r['us']; its += int(r['iterations']); xs_plan = r['xs']
  g, obj, X = true_eval(us)
  if xs_plan is None: xs_plan = np.asarray(X)
  inc = dict(us=us, goal=g, obj=obj, X=X)
  if project == 'lqr':
    Xp, Up = lqr_project(xs_plan, us, 1.0)
    gp, op, _ = true_eval(Up)
    if gp < g:
      inc = dict(us=Up, goal=gp, obj=op, X=Xp); g, obj = gp, op
  m_of = lambda g_, o_: g_ if merit_kind == 'goal' else o_
  hist = [dict(burst=0, sm=None, k=0 if disc_mode == 'none' else disc_iters, goal=g, obj=obj, acc=True)]
  print(f'seed {seed} discovery true goal {g:.4f} obj {obj:.3g}', flush=True)
  levels = ([max(ladder) * 10 ** i for i in range(up_ok - 1, 0, -1)] if up_ok > 1 else []) + ladder
  li = levels.index(ladder[0]); k = k0; last_ok = li; max_sm = ladder[0]
  used = 0
  best_seen = g
  last_it = None
  n_rej = 0
  n_post = 0   # bursts run since the incumbent first went below 0.01 (A5: --post-ok caps them)
  n_ext = 0; max_ext = int(os.environ.get('MAX_EXT', '4'))
  ncyc = int(os.environ.get('NCYC', '0')); cyc = 0; c0 = 1; near = False; wide = 1.0; cyc_goal0 = inc['goal']; cyc_log = []
  while True:
    if not (used < budget and inc['goal'] > stop_tol and k >= kmin and n_post < post_ok):
      # E0 fix 1: a budget stop during steady improvement (last two accepted bursts of this cycle each gained >= 2 %)
      # is not a stall: extend the cycle at the current level (bounded by --max-ext extensions per seed)
      gains = [h_['gain'] for h_ in hist[c0:] if h_.get('acc') and 'gain' in h_]
      if used >= budget and inc['goal'] > stop_tol and k >= kmin and n_post < post_ok and len(gains) >= 2 \
         and min(gains[-2:]) >= 0.02 and n_ext < max_ext:
        n_ext += 1; used = 0
        print(f'  == extend cycle {cyc} (gains {gains[-2:]})', flush=True)
        continue
      cyc_log.append(dict(cyc=cyc, start=cyc_goal0, end=inc['goal'], level=levels[c_li] if cyc else ladder[0], near=near))
      if inc['goal'] <= stop_tol or n_post >= post_ok or cyc >= ncyc:
        break
      improved = inc['goal'] < 0.9 * cyc_goal0
      if inc['goal'] < 0.03:
        if near and not improved: break
        near = True; li = levels.index(0.1)
      elif improved:
        near = False; wide = 1.0; li = levels.index(1.0)
      elif wide < 10:
        near = False; wide = 10.0; li = levels.index(10.0)
      else:
        break
      c_li = li
      if os.environ.get('SAVE_DIR'):   # E0 fix 2: cycle-start incumbents (inputs of E6)
        os.makedirs(os.environ['SAVE_DIR'] + '/cycstart', exist_ok=True)
        np.savez(f"{os.environ['SAVE_DIR']}/cycstart/{label}_{name}_{seed}_c{cyc + 1}.npz", us=inc['us'], goal=inc['goal'], level=levels[li])
      cyc += 1; k = k0; used = 0; n_rej = 0; last_it = None; c0 = len(hist); last_ok = li; cyc_goal0 = inc['goal']
      print(f'  == cycle {cyc}: goal {inc["goal"]:.4f} start level {levels[li]} near {near}', flush=True)
    sm = levels[li]; max_sm = max(max_sm, sm)
    t0_ = time.perf_counter()
    rc_ = RC(); rc_.__enter__()
    eps = jp.asarray(trans.contact_smoothing * sm)
    use_iter = (start == 'iterate' and last_it is not None) or (start == 'hybrid' and last_it is not None and n_rej > 0)
    if use_iter:   # continue from the last burst's iterate (fresh duals/barrier, no projection onto a rollout)
      Xs, w = last_it
    elif (start == 'true' or (start == 'hybrid' and project == 'lqr')) and np.isfinite(inc['X']).all():
      Xs = np.asarray(inc['X']); cfs = np.asarray(law_forces(jp.asarray(Xs[:-1]), jp.asarray(inc['us']), eps))
    else:
      Xs, cfs = law_roll(jp.asarray(task.x_0), jp.asarray(inc['us']), eps)
      Xs, cfs = np.asarray(Xs), np.asarray(cfs)
    if not use_iter:
      w = np.concatenate([inc['us'], cfs], 1)
    if not np.isfinite(Xs).all():
      Xs = np.tile(np.asarray(task.x_0), (N + 1, 1)); w = np.concatenate([inc['us'], np.zeros((N, ncf))], 1)
    rc_.__exit__(None, None, None)
    TM['retr'] += time.perf_counter() - t0_
    boosted = qf_boost != 1.0 and inc['goal'] < boost_below
    if boosted:
      import dataclasses
      p0 = trans.params
      trans.params = dataclasses.replace(p0, Qf=p0.Qf * qf_boost)
      if form == 'B2': E.params = trans.params
    any_acc = (near or any(h_['acc'] for h_ in hist[c0:]))
    t0_ = time.perf_counter()
    S = solve_ph(dict(PH(sm, mu_burst, 1e-4), hess=hess_after) if any_acc else PH(sm, mu0 if disc_mode == 'none' else mu_burst, disc_mumin), Xs, w, k)
    last_it = (np.array(S.cur.x), np.array(S.cur.w))
    TM['solve'] += time.perf_counter() - t0_
    if boosted:
      trans.params = p0
      if form == 'B2': E.params = p0
    it = int(S.it); its += it; used += max(it, 1)
    us_c = np.array(S.cur.w)[:, :nu]
    gc, oc, Xc = true_eval(us_c)
    proj_used = False
    if project == 'lqr':
      t0_ = time.perf_counter()
      Xp, Up = lqr_project(np.array(S.cur.x), us_c, sm)
      TM['lqr'] += time.perf_counter() - t0_
      gp, op, _ = true_eval(Up)
      if m_of(gp, op) < m_of(gc, oc):
        us_c, gc, oc, Xc, proj_used = Up, gp, op, Xp, True
    acc = True if accept_all else m_of(gc, oc) < m_of(inc['goal'], inc['obj'])
    if acc and disc_frac > 0 and not any_acc and k < 40:
      acc = gc < (1.0 - disc_frac) * inc['goal']   # a weak first improvement does not end discovery
    best_seen = min(best_seen, gc)
    h = dict(burst=len(hist), sm=sm, k=k, it=it, status=ippas.STATUS_NAMES.get(int(S.status), str(int(S.status))),
             plan_goal=sc(np.asarray(S.cur.x)[-1]), goal=gc, obj=oc, acc=bool(acc), boosted=bool(boosted), proj=bool(proj_used))
    hist.append(h)
    print('   ', h, flush=True)
    if inc['goal'] < 0.01:
      n_post += 1
    if acc:
      h['gain'] = 1.0 - gc / inc['goal'] if np.isfinite(inc['goal']) and inc['goal'] > 0 else 1.0
      inc = dict(us=us_c, goal=gc, obj=oc, X=Xc)
      last_ok = li
      li = min(li + 1, len(levels) - 1)   # accepted: next burst one level lower
      n_rej = 0
      k = min(k, k0)
    elif start == 'hybrid' and not (near or any(h_['acc'] for h_ in hist[c0:])) and k < (40 if descend else 100):
      n_rej += 1; k = min(2 * k, 100)   # nothing accepted yet: a lazy discovery, continue from the iterate with a doubled cap
    elif descend and not (near or any(h_['acc'] for h_ in hist[c0:])) and li < len(levels) - 1:
      li += 1; k = k0; n_rej = 0        # still nothing accepted: next lower smoothing, restart from the incumbent's rollout
    elif start == 'hybrid' and n_rej < 2:
      n_rej += 1     # hybrid: keep iterating from the burst's own iterate at the same level and k
    else:
      n_rej = 0 if start == 'hybrid' else n_rej
      if adapt:
        li = max(min(last_ok, li) - 1, 0) if li <= last_ok else last_ok
        k = k // 2
      else:
        li = min(li + 1, len(levels) - 1)   # fixed ladder control: walk down regardless
  t = time.perf_counter() - tic
  row = dict(task=name, label=label, seed=seed, true_goal=inc['goal'], true_ok=bool(inc['goal'] < 0.01),
             disc_goal=hist[0]['goal'], n_acc=sum(h['acc'] for h in hist[1:]), n_bursts=len(hist) - 1, cycles=cyc, cyc_log=cyc_log,
             max_smoothing=max([max_sm] + [q.get('smoothing', 1) for q in (disc_sched or [])]), best_seen=best_seen, n_ext=n_ext, it=its, time=t, timing={k_: round(v_, 3) for k_, v_ in TM.items()}, hist=hist,
             cfg=dict(disc_frac=disc_frac, roll_cpu=roll_cpu, post_ok=post_ok, stop_tol=stop_tol, kmin=kmin, hess_after=hess_after, mu0=mu0, k=k0, budget=budget, ladder=ladder, merit=merit_kind, adapt=adapt, mu_burst=mu_burst, accept_all=accept_all, form=form, qf_boost=qf_boost, project=project, start=start, disc_mode=disc_mode, disc_mumin=disc_mumin, descend=descend))
  print(json.dumps({k_: v for k_, v in row.items() if k_ != 'hist'}, default=J), flush=True)
  with open(out, 'a') as fo:
    fo.write(json.dumps(row, default=J) + '\n')
  if os.environ.get('SAVE_DIR'):
    os.makedirs(os.environ['SAVE_DIR'], exist_ok=True)
    np.savez(f"{os.environ['SAVE_DIR']}/{label}_{name}_{seed}.npz", us=inc['us'])
