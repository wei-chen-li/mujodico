"""LFDDP: feasibility-driven DDP on the lifted LMS variables (x, u, cf), with model management.

Problem (the same NLP IP-PAS solves; LMS transcription of the wrapper):
    min  sum_k l(x_k, u_k) + l_N(x_N)
    s.t. x_{k+1} = f(x_k, u_k, cf_k)                    (dynamics given the contact forces)
         h(x_k, u_k, cf_k) = cf_k - Phi_s(x_k, u_k) = 0  (lifted contact consistency, smoothing s)
         u in [u_lb, u_ub];  |qvel| <= v_max (exterior quadratic penalty, as the IP's state bound)

Iterate: (X, U, CF) with gaps  fbar_{k+1} = f(x_k, u_k, cf_k) - x_{k+1},  hbar_k = h(x_k, u_k, cf_k).

Backward pass (Riccati / DDP, Gauss-Newton or with lagged dynamics curvature): since dh/dcf = I the
contact-force step is eliminated exactly, dcf = Phi_x dx + Phi_u du - hbar, leaving
    dx' = A dx + B du + a,   A = f_x + f_cf Phi_x,  B = f_u + f_cf Phi_u,  a = fbar - f_cf hbar
and a box-constrained QP per stage for du (projected Newton on the free set), Levenberg-Marquardt
regularization on Quu.

Forward pass (FDDP nonlinear rollout with gap contraction):
    u^ = clip(u + alpha k + K (x^ - x)),  cf^ = Phi_s(x^, u^) + (1-alpha) hbar,
    x^' = f(x^, u^, cf^) - (1-alpha) fbar
From a consistent start every iterate is an exact trajectory of the smoothed law.

Model management (as PIP): every iteration the new controls are read out under the TRUE law (CPU), open loop
and with the DDP feedback gains; the best is the incumbent.  When the smoothing level is solved (expected
decrease below tol) or progress stalls, change the model: sharpen s by 10 if the smoothing error at the
incumbent |J_s(u*) - J_smin(u*)| > kappa r*, else terminal weight x10 (<= --boosts), then continue from the
incumbent's controls re-simulated at the new level.

usage: lfddp.py <Task> <a:b> <label> <out.jsonl> [--sig0 1] [--sig-min 1e-3] [--kappa 0.25] [--stall 20]
       [--budget 250] [--stop-tol 3e-3] [--boosts 2] [--vel-w 1e3] [--curv 0|1] [--readout open|fb]
       [--lin-dev gpu|cpu] [--bf-dev cpu|gpu] [--tol-d 1e-4]"""
import os, sys, json, time, dataclasses, threading
_PLOCK = threading.Lock()
if '--par-ls' in sys.argv and sys.argv[sys.argv.index('--par-ls') + 1] != '0':
  # parallel line search: several CPU "devices" (one per step size), set before jax is imported
  os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=16').strip()
import numpy as np
import jax, jax.numpy as jp
H_ = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs'
sys.path.insert(0, H_ + '/handoff_2026-11-22/scripts')
sys.path.insert(0, H_ + '/handoff_2026-11-14/followup/scripts')
if '--face-ff' in sys.argv:   # planning-only per-face far-field contact candidates (trajopt/_src/face_ff.py)
  from mujodico.trajopt._src import transcriptions as _trx_ff
  _trx_ff.FACE_FF = dict(weight=float(sys.argv[sys.argv.index('--face-ff') + 1]))
from mujodico.examples.benchmark_ippas import make_task
from mujodico.examples.methods import IppasLiftedMultipleShooting, util
import hard_tasks  # noqa
import extra_tasks  # noqa
from reloc_lms import LMS_KW


def _vjf(fun, argnums):
  """value and forward-mode Jacobian(s)."""
  def g(*a):
    J, y = jax.jacfwd(lambda *b: (fun(*b), fun(*b)), argnums=argnums, has_aux=True)(*a)
    return y, J
  return g


def _boxqp(H, g, Hx, lo, hi, iters=4):
  """min 1/2 k'Hk + g'k s.t. lo <= k <= hi (projected Newton); gains of the free set."""
  n = g.shape[0]
  I = jp.eye(n)
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


class LFDDP:

  def __init__(self, task, lms, vel_bound=5.0, vel_w=1e3, curv=False, clear_w=None, clear_margin=0.0):
    trans = lms.transcription
    self.trans = trans
    m = task.model
    self.nq, self.nv, self.nu, self.N, self.nx = m.nq, m.nv, task.nu, task.horizon, task.nx
    nu = self.nu; nq, nv = self.nq, self.nv
    self.ncf = ncf = trans.nw - nu
    self.base = float(trans.contact_smoothing)
    lb, ub = (np.asarray(b_, float) for b_ in task.u_bounds) if task.u_bounds is not None else (np.full(nu, -np.inf), np.full(nu, np.inf))
    self.lb, self.ub = jp.asarray(lb), jp.asarray(ub)
    cf_lb = np.asarray(trans.w_bounds[0], float)[nu:]
    self.bmask = jp.asarray(np.isfinite(cf_lb))
    # normal force of each contact as a linear map of the lifted forces: pyramidal = sum of its edge forces,
    # elliptic = the normal component, frictionless = the single component
    from mujodico.trajopt._src import transcriptions as trx
    mm = trx.mjx.put_model(m, impl='jax', contact_approximation=task.contact_approx, contact_smoothing=task.contact_smoothing)
    _, contacts = trx._extract_contact_indices(mm)
    G = np.zeros((len(contacts), ncf))
    for i, ct in enumerate(contacts):
      if ct.type == trx._Contact.Type.PYRAMIDAL:
        G[i, ct.cf_index:ct.cf_index + 2 * (ct.dim - 1)] = 1.0
      else:
        G[i, ct.cf_index] = 1.0
    self.G = G = jp.asarray(G)
    self.ncontact = G.shape[0]
    NFLOOR = 1e-30
    f = trans.dynamics
    law = lambda x, u, eps: -trans.stage_equality(x, jp.concatenate([u, jp.zeros(ncf)]), None, eps)
    self.law = law

    clear = getattr(lms, '_clear_x0', None)     # link-object clearance g(q) >= 0 of the task (None if absent)
    cspec = getattr(lms._build_task, 'clearance_spec', None) or {}
    cw = float(clear_w if clear_w is not None else cspec.get('terminal_weight', 1e5))
    self.has_clear = clear is not None

    def vpen(x):
      v = x[nq:nq + nv]
      pen = vel_w * jp.sum(jp.maximum(jp.abs(v) - vel_bound, 0.0) ** 2)
      if clear is not None:
        pen = pen + cw * jp.sum(jp.minimum(clear(x) - clear_margin, 0.0) ** 2)
      return pen

    def lst(x, u, P):
      return trans.stage_cost(x, jp.concatenate([u, jp.zeros(ncf)]), P) + vpen(x)

    def lterm(x, P):
      return trans.terminal_cost(x, P) + vpen(x)
    self.lst, self.lterm = lst, lterm

    def bar(c, mu):   # log barrier on each contact's normal force (contact attraction on the far-field tail)
      n = G @ c
      return -mu * jp.sum(jp.log(jp.maximum(n, NFLOOR)))
    self.bar = bar

    def lin(X, U, C, eps, P, lam, mu):
      def stage(x, u, c, xn, lm):
        w = jp.concatenate([u, c])
        fv, (fx, fw) = _vjf(f, (0, 1))(x, w)
        lv, (Lx, Lu) = _vjf(law, (0, 1))(x, u, eps)
        fu, fc = fw[:, :nu], fw[:, nu:]
        A = fx + fc @ Lx; B = fu + fc @ Lu
        fbar = fv - xn; hbar = c - lv
        a = fbar - fc @ hbar
        lx, lu = jax.grad(lst, (0, 1))(x, u, P)
        (lxx, lxu), (lux, luu) = jax.hessian(lst, (0, 1))(x, u, P)
        # barrier in the lifted variable, mapped through the linearized consistency dcf = Lx dx + Lu du - hbar
        n = G @ c; live = n > NFLOOR; nb = jp.where(live, n, 1.0)
        gn = jp.where(live, -mu / nb, 0.0); Hn = jp.where(live, mu / nb ** 2, 0.0)
        gne = gn - Hn * (G @ hbar)
        Nx, Nu = G @ Lx, G @ Lu
        lx = lx + Nx.T @ gne; lu = lu + Nu.T @ gne
        lxx = lxx + Nx.T @ (Hn[:, None] * Nx); lux = lux + Nu.T @ (Hn[:, None] * Nx); luu = luu + Nu.T @ (Hn[:, None] * Nu)
        if curv:   # lagged dynamics curvature: Hessian of lam' F(x, u) with F the composite smoothed step
          Fc = lambda x_, u_: lm @ f(x_, jp.concatenate([u_, law(x_, u_, eps)]))
          (cxx, cxu), (cux, cuu) = jax.hessian(Fc, (0, 1))(x, u)
          lxx = lxx + cxx; lux = lux + cux; luu = luu + cuu
        return A, B, a, lx, lu, lxx, lux, luu, fbar, hbar
      out = jax.vmap(stage)(X[:-1], U, C, X[1:], lam)
      return out + (jax.grad(lterm)(X[-1], P), jax.hessian(lterm)(X[-1], P))
    self._lin = jax.jit(lin)

    def backward(L, U, reg):
      A, B, a, lx, lu, lxx, lux, luu, fbar, hbar, VxN, VxxN = L
      I = jp.eye(nu)
      def step(carry, inp):
        Vx, Vxx = carry
        A_, B_, a_, lx_, lu_, lxx_, lux_, luu_, u_ = inp
        Vx_ = Vx + Vxx @ a_
        Qx = lx_ + A_.T @ Vx_; Qu = lu_ + B_.T @ Vx_
        Qxx = lxx_ + A_.T @ Vxx @ A_; Quu = luu_ + B_.T @ Vxx @ B_; Qux = lux_ + B_.T @ Vxx @ A_
        Qr = 0.5 * (Quu + Quu.T) + reg * I
        ok = jp.all(jp.isfinite(jp.linalg.cholesky(Qr)))
        k, K = _boxqp(Qr, Qu, Qux, self.lb - u_, self.ub - u_)
        Vx_n = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
        Vxx_n = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
        Vxx_n = 0.5 * (Vxx_n + Vxx_n.T)
        return (Vx_n, Vxx_n), (k, K, k @ Qu, k @ Quu @ k, ok, Vx_)
      (Vx0, _), (k, K, d1, d2, ok, lam) = jax.lax.scan(step, (VxN, VxxN), (A, B, a, lx, lu, lxx, lux, luu, U), reverse=True)
      return k, K, jp.sum(d1), jp.sum(d2), jp.all(ok), lam
    self._bwd = jax.jit(backward)

    def forward(X, U, C, k, K, fbar, hbar, alpha, eps, x0, P, mu, kscale=0.0):
      fb_next = fbar            # fbar[k] is the gap at x_{k+1}
      def step(xh, inp):
        x, u, kk, KK, fb, hb = inp
        uh = jp.clip(u + alpha * kk + jp.where(kscale > 0, alpha, 1.0) * (KK @ (xh - x)), self.lb, self.ub)
        ch = law(xh, uh, eps) + (1.0 - alpha) * hb
        xn = f(xh, jp.concatenate([uh, ch])) - (1.0 - alpha) * fb
        return xn, (xh, uh, ch)
      xN, (Xh, Uh, Ch) = jax.lax.scan(step, x0, (X[:-1], U, k, K, fb_next, hbar))
      Xh = jp.concatenate([Xh, xN[None]])
      J = jnp_cost(Xh, Uh, Ch, P, mu)
      ok = jp.all(jp.isfinite(Xh))
      return Xh, Uh, Ch, jp.where(ok, J, jp.inf)

    def jnp_cost(X, U, C, P, mu):
      return (jp.sum(jax.vmap(lst, (0, 0, None))(X[:-1], U, P)) + lterm(X[-1], P) +
              jp.sum(jax.vmap(bar, (0, None))(C, mu)))
    self._fwd = jax.jit(forward)
    # all line-search step sizes in one call (one batched rollout instead of up to five sequential ones)
    self._fwd_b = jax.jit(jax.vmap(forward, in_axes=(None,) * 7 + (0,) + (None,) * 5))
    cpus = jax.devices('cpu')
    self.ncpu = len(cpus)
    # (pmap over CPU devices segfaults in XLA's compiler for the SAP while-loops; the driver dispatches
    #  single-device calls on separate CPU devices concurrently instead)
    self._cost = jax.jit(jnp_cost)

    def rollout(x0, U, eps):
      def step(x, u):
        c = law(x, u, eps)
        return f(x, jp.concatenate([u, c])), (x, c)
      xN, (X, C) = jax.lax.scan(step, x0, U)
      return jp.concatenate([X, xN[None]]), C
    self._roll = jax.jit(rollout)
    self._forces = jax.jit(jax.vmap(law, in_axes=(0, 0, None)))
    def tvlqr(A, B, Qd, R, PN):   # time-varying LQR gains about a plan (F5's projection weights)
      def step(Pm, inp):
        A_, B_ = inp
        K_ = -jp.linalg.solve(R + B_.T @ Pm @ B_, B_.T @ Pm @ A_)
        Pn = Qd + A_.T @ Pm @ (A_ + B_ @ K_)
        return 0.5 * (Pn + Pn.T), K_
      _, K = jax.lax.scan(step, PN, (A, B), reverse=True)
      return K
    self._tvlqr = jax.jit(tvlqr)

    def track_roll(x0, Xr, Ur, K, eps):   # closed-loop rollout of the smoothed law tracking (Xr, Ur) with gains K
      def step(x, inp):
        xr, ur, KK = inp
        e = eps
        u = jp.clip(ur + KK @ (x - xr), self.lb, self.ub)
        c = law(x, u, e)
        return f(x, jp.concatenate([u, c])), (x, u, c)
      xN, (X, U, C) = jax.lax.scan(step, x0, (Xr[:-1], Ur, K))
      return jp.concatenate([X, xN[None]]), U, C
    self._track = jax.jit(track_roll)


if __name__ == '__main__':
  name, rng_, label, out = sys.argv[1:5]
  args = sys.argv[5:]
  # the LAST occurrence wins, so that flags appended to a launcher's defaults override them
  opt = lambda k, d, f=float: f(args[len(args) - 1 - args[::-1].index(k) + 1]) if k in args else d
  sig0 = opt('--sig0', 1.0); sig_min = opt('--sig-min', 1e-3); kappa = opt('--kappa', 0.25)
  stall = opt('--stall', 20, int); budget = opt('--budget', 250, int); stop_tol = opt('--stop-tol', 3e-3)
  max_boosts = opt('--boosts', 2, int); vel_w = opt('--vel-w', 1e3); curv = bool(opt('--curv', 0, int))
  readout = opt('--readout', 'fb', str); tol_d = opt('--tol-d', 1e-4)
  lin_dev = opt('--lin-dev', 'gpu', str); bf_dev = opt('--bf-dev', 'cpu', str)
  reg0 = opt('--reg0', 1e-6); verbose = opt('--verbose', 0, int)
  mu0 = opt('--mu0', 1e-2); mu_fac = opt('--mu-fac', 0.1)
  # (1) step control
  lm = opt('--lm', 0, int)            # regularization follows the accepted step size (Levenberg-Marquardt)
  kscale = opt('--kscale', 0.0)       # 1: scale the feedback term by alpha too (diagnostic)
  # (2) smoothing of the FDDP phase, ratio test
  sig_ddp = opt('--sig-ddp', 0.0)     # >0: smoothing of the FDDP phase after a warm start (fixed warm start)
  mistrust = opt('--mistrust', 0, int); tau = opt('--tau', 0.05); eta = opt('--eta', 0.25)
  dwell = opt('--dwell', 0, int)              # FDDP phase: iterations at a model level before the ratio test may fire
  handoff_mucut = opt('--handoff-mucut', 0, int)   # 1: also cut the barrier when the hand-off sharpens
  # (3) warm start
  init_plan = opt('--init-plan', None, str)   # npz with xs, us, cs: start LFDDP from this (lifted, possibly gapped) plan
  warm = 'init' if init_plan else opt('--warm', 'fixed' if '--warm-ip' in args else 'none', str)   # none | fixed | pip | init
  warm_ip = opt('--warm-ip', 20, int)  # fixed: IP iterations; pip: cap on IP iterations before a forced hand-off
  ip_mu = opt('--ip-mu', 1e-3); ip_mumin = opt('--ip-mumin', 1e-2); ip_stall = opt('--ip-stall', 20, int)
  handoff_from = opt('--handoff-from', 'iterate', str)   # iterate (gapped IP iterate) | incumbent | true
  restart = opt('--restart', 'sim', str)   # model-change restarts: sim (smoothed simulation of the incumbent's
                                           # controls) | true (incumbent's TRUE states, smoothed-law forces; gaps)
  cycle = opt('--cycle', 0, int)           # >0: when LFDDP exhausts its model changes short of the goal, re-explore
                                           # with the lifted IP from the incumbent (at most this many times)
  ip_cycle = opt('--ip-cycle', 40, int); total_budget = opt('--total-budget', 400, int)
  # fidelity cycles (two-sided model management): on exhaustion, re-explore from the incumbent at a COARSE level
  vcyc = opt('--vcycle', 0, int)            # 1: the cycle explores at sig0 * 10**j (j = consecutive unproductive cycles)
  v_up = opt('--v-up', 1, int)
  v_top = opt('--v-top', 0, int)            # tr: the trust region starts at sig0 * 10**v_top              # largest j (1: levels sig0 and 10 sig0)
  v_cap = opt('--v-cap', -1, int)           # IP cap of a coarse excursion (default warm_ip, as the first exploration)
  v_stall = opt('--v-stall', -1, int)       # IP stall count of a coarse excursion (default ip_stall)
  v_guard = opt('--v-guard', 1, int)        # ratio-test discovery guard per excursion (true law responded in THIS excursion)
  v_budget = opt('--v-budget', 0, int)      # extra iteration budget per cycle (added to total_budget)
  ip_ro = opt('--ip-readout', 'open', str)  # IP read-outs: open (open-loop replay of the IP controls) | lqr (also
                                            # TVLQR-tracked under the true law about the IP iterate; best of both)
  ip_ro_scope = opt('--ip-ro-scope', 'cycle', str)   # cycle: only in fidelity-cycle excursions | all
  v_level = opt('--v-level', 'fixed', str)    # fixed: cycle level sig0 * 10**j | acc: L* * 10**j, L* = the coarsest
                                             # level whose smoothing error at the incumbent is within kappa r* (the
                                             # sharpening test of the ladder, read in reverse)
  v_reset = opt('--v-reset', 'prod', str)    # prod: restart the model ladder at the cycle level only after a productive
                                             # excursion | always: a cycle = PLF's ladder re-run from the incumbent at sig_c
  prog_stall = opt('--prog-stall', 0, int)   # LFDDP stall = no tau-relative decrease of r* within --stall iterations
                                             # (0: any decrease resets the count; 1: after the first fidelity cycle; 2: always)
  v_bud_cycle = opt('--v-bud-cycle', 0, int)  # 1: the LFDDP budget ending short of the goal triggers a fidelity cycle
  v_restart = opt('--v-restart', None, str)  # start point of a coarse excursion: sim | true | auto (default: --restart)
  from_us = opt('--from-us', None, str)      # npz pattern with {seed}: start from these controls (the first exploration
                                             # then is a coarse excursion from that plan)
  nu_merit = opt('--nu-merit', 1e3)
  lpec = opt('--lpec', 0, int)          # >0: at most this many LPEC branch-step rounds when PLF-V ends short of the goal
  lpec_d0 = opt('--lpec-delta', 0.05)   # LPEC trust radius (controls, inf-norm); /2.5 after a step without tau-decrease (<= 2 in a row)
  log_steps = opt('--log-steps', 0, int)   # 1: log (alpha, rho, reg) of every accepted feasible-phase step
  ls_spec = opt('--ls-spec', 0, int)   # 1: speculative LFDDP attempts (backward at reg and 10 reg, both line searches at once)
  lpec_solver = opt('--lpec-solver', 'grb', str)   # grb: Gurobi MILP + QP | fast: Gurobi MILP + structured IP QP (lpec_fast)
  lpec_par = opt('--lpec-par', 1, int)   # >1: the next k trust radii of the LPEC shrink sequence run as concurrent rounds
  lpec_start = opt('--lpec-start', 'coarse', str)   # coarse: an LPEC round starts with a coarse excursion | fine: LFDDP at sig_min
  v_local = opt('--v-local', 0, int)   # 1: fidelity-cycle excursions coarsen only the incumbent's contact-free stages (LFDDP)
  ro_overlap = opt('--ro-overlap', 0, int)   # 1: IP read-outs overlap the next IP chunk (same iterates)
  ip_hess = opt('--ip-hess', 1.0)        # lifted IP dynamics-curvature weight (0: Gauss-Newton on the dynamics)
  ip_hess_cyc = opt('--ip-hess-cycle', None)   # the same, in fidelity-cycle excursions only (default: --ip-hess)
  # (4) read-outs, (5) line search
  ro_every = opt('--ro-every', 1, int)
  par_ls = opt('--par-ls', 0, int)
  ls_mode = 'par' if par_ls else opt('--ls', 'batch', str)   # seq | batch | par
  cfg = {k_: v_ for k_, v_ in dict(locals()).items() if k_ in (
      'sig0', 'sig_min', 'kappa', 'stall', 'budget', 'stop_tol', 'max_boosts', 'vel_w', 'curv', 'readout', 'tol_d',
      'lin_dev', 'bf_dev', 'reg0', 'mu0', 'mu_fac', 'lm', 'kscale', 'sig_ddp', 'mistrust', 'tau', 'eta', 'warm',
      'warm_ip', 'ip_mu', 'ip_mumin', 'ip_stall', 'handoff_from', 'nu_merit', 'ro_every', 'ls_mode', 'dwell',
      'handoff_mucut', 'restart', 'cycle', 'ip_cycle', 'total_budget', 'vcyc', 'v_up', 'v_cap', 'v_stall', 'v_guard', 'v_budget', 'ip_ro', 'ip_ro_scope', 'v_restart', 'from_us', 'v_level', 'v_reset', 'prog_stall', 'v_bud_cycle', 'v_top', 'ip_hess', 'lpec', 'lpec_d0', 'ls_spec', 'lpec_solver', 'lpec_par', 'lpec_start', 'ro_overlap', 'v_local', 'ip_hess_cyc')}
  a, b = (int(v) for v in rng_.split(':'))
  CPUS = jax.devices('cpu'); CPU = CPUS[0]; GPU = jax.devices()[0]
  DL = GPU if lin_dev == 'gpu' else CPU; DB = GPU if bf_dev == 'gpu' else CPU
  toL = lambda t: jax.device_put(t, DL); toB = lambda t: jax.device_put(t, DB); toC = lambda t: jax.device_put(t, CPU)
  toG = lambda t: jax.device_put(t, GPU)
  ALPHAS = tuple(float(v) for v in opt('--alphas', '1,0.5,0.25,0.1,0.03', str).split(','))
  _LMF = {1.0: 0.25, 0.5: 1.0, 0.25: 3.0, 0.1: 10.0, 0.03: 30.0}
  LMF = {a_: _LMF.get(a_, 0.25 if a_ >= 1.0 else min(30.0, 0.75 / a_)) for a_ in ALPHAS}
  S = None; SL = None
  for seed in range(a, b):
    task = make_task(name, seed)
    if S is None:
      PHip = dict(smoothing=sig0, iters=10, hess=[ip_hess, 0, 0], mu_init=ip_mu, mu_min=ip_mumin)
      lms = IppasLiftedMultipleShooting(task, smoothing_schedule=[PHip], max_iter=(max(10, warm_ip + 10) if v_guard < 3 else total_budget), **LMS_KW)
      truem = IppasLiftedMultipleShooting(task)
      S = LFDDP(task, lms, vel_w=vel_w, curv=curv, clear_w=opt('--clear-w', None, float),
                clear_margin=opt('--clear-margin', 0.0))
      trans = lms.transcription
      qidx = list(task.quaternion_w_idxs)
      if warm == 'pip':
        from hs_export import ExportedHomotopyIpPas   # HomotopyIpPas with an on-disk export of its traced loop
        HS = ExportedHomotopyIpPas(lms.solver)
      truem.rollout(task.x_0, np.zeros((task.horizon, task.nu)), 0.0)
      _troll = truem._rollout
      _true_step = truem.transcription.true_step
      _sc = lambda Q, g, x: jp.where(jp.all(jp.isfinite(x)), util.state_cost(Q, g, x, qidx), jp.inf)
      @jax.jit
      def true_eval(Q, g, x0, us):
        X, _ = _troll(x0, us, jp.asarray(0.0))
        return _sc(Q, g, X[-1])
      def _true_one(s, Q, g, Xp, Up, K):
        """true-law rollout with gains s*K about the plan (s = 0: open loop)"""
        def body(x, k):
          u = jp.clip(Up[k] + s * (K[k] @ (x - Xp[k])), S.lb, S.ub)
          xn, _ = _true_step(x, u, jp.asarray(0.0))
          return xn, u
        xN, Ua = jax.lax.scan(body, Xp[0], jp.arange(Up.shape[0]))
        return _sc(Q, g, xN), Ua
      true_both = jax.jit(jax.vmap(_true_one, in_axes=(0, None, None, None, None, None)))   # fused: open + feedback
      true_one = jax.jit(_true_one)
      par_ro = ls_mode == 'par' and len(CPUS) >= 7
      plan_goal = jax.jit(lambda Q, g, X: _sc(Q, g, X[-1]))
      if os.environ.get('PLF_FN_EXPORT', '1') != '0' and getattr(lms.solver, 'export_key', None) and lms.solver.export_dir:
        # serve the driver's jitted functions from on-disk exports (tracing/lowering once across processes)
        import hashlib
        from hs_export import ExportedJit
        _fk = hashlib.sha256('|'.join((lms.solver.export_key, getattr(truem.solver, 'export_key', '') or '',
                                       hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
                                       repr((vel_w, curv, opt('--clear-w', None, float), opt('--clear-margin', 0.0), qidx)))
                                      ).encode()).hexdigest()
        _xd = lms.solver.export_dir
        for _n in ('_lin', '_bwd', '_fwd', '_fwd_b', '_cost', '_roll', '_forces', '_tvlqr', '_track'):
          setattr(S, _n, ExportedJit(getattr(S, _n), 'S' + _n, _fk, _xd))
        true_eval = ExportedJit(true_eval, 'true_eval', _fk, _xd); true_both = ExportedJit(true_both, 'true_both', _fk, _xd)
        true_one = ExportedJit(true_one, 'true_one', _fk, _xd); plan_goal = ExportedJit(plan_goal, 'plan_goal', _fk, _xd)
    else:
      lms.update(task); truem.update(task)
    Qm = jp.asarray(getattr(task, 'metric_Q', task.Q)); g0 = jp.asarray(task.x_goal); x0 = jp.asarray(task.x_0)
    QmC, g0C, x0C = toC(Qm), toC(g0), toC(x0); x0B = toB(x0); QmB, g0B = toB(Qm), toB(g0)
    N, nu, nx = S.N, S.nu, S.nx
    p_base = trans.params
    Pof = lambda boosts: dataclasses.replace(p_base, Qf=p_base.Qf * 10.0 ** boosts) if boosts else p_base
    T = dict(ip=0.0, lin=0.0, bwd=0.0, fwd=0.0, true=0.0, xfer=0.0, ro=0.0, n_bwd=0, n_ls=0)
    def tg(us):
      t0 = time.perf_counter(); r = float(true_eval(QmC, g0C, x0C, toC(us))); T['true'] += time.perf_counter() - t0
      return r
    def sim(us, sig):
      return S._roll(x0B, toB(us), toB(jp.asarray(S.base * sig)))
    def sgoal(us, sig):
      return float(plan_goal(QmB, g0B, sim(us, sig)[0]))
    tic = time.perf_counter()
    U_from = None if from_us is None else np.asarray(np.load(from_us.format(seed=seed))['us'])
    lp_log = []; lp_D = lpec_d0; lp_shrink = 0; best = None; tot_it = 0; tot_ip = 0; lp_step = None
    def run_round(U_from, lpec_round):   # one PLF-V solve; round 0 from scratch (or --from-us), LPEC rounds from a step
      sig = sig0; boosts = 0; mu = mu0
      U = toB(jp.zeros((N, nu))) if U_from is None else toB(jp.asarray(U_from))
      X, C = sim(U, sig)
      u_inc = U; r_inc = tg(U); r0 = r_inc; its_ok = None; t_ok = None
      ip_its = 0; handoff = None; events = []; streams = []; cyc_log = []; jc = 0; r_cyc = None
      def restart_point(us, sig_, mode):
        """sim: smoothed-law simulation of us; true: the TRUE-law states of us with smoothed-law forces (consistent
        per stage; dynamics gaps where the smoothed and true laws differ, which FDDP / the IP close)."""
        if mode == 'auto':   # true states only if the smoothed simulation does not reproduce the incumbent's outcome
          Xs_, Cs_ = sim(us, sig_)
          js_ = float(plan_goal(QmB, g0B, Xs_))
          if abs(js_ - r_inc) <= kappa * max(r_inc, 1e-3):
            return Xs_, Cs_
          mode = 'true'
        if mode == 'true':
          Xt = toB(truem._rollout(x0C, toC(us), jp.asarray(0.0))[0])
          if bool(jp.all(jp.isfinite(Xt))):
            return Xt, S._forces(Xt[:-1], toB(us), toB(jp.asarray(S.base * sig_)))
        return sim(us, sig_)

      def start_point(Sip, sig_, mode):
        if mode == 'iterate' and Sip is not None:
          Wi = np.asarray(Sip.cur.w)
          return toB(jp.asarray(np.asarray(Sip.cur.x))), toB(jp.asarray(Wi[:, :nu])), toB(jp.asarray(Wi[:, nu:]))
        Xr, Cr = restart_point(u_inc, sig_, mode if mode in ('true', 'auto') else 'sim')
        return Xr, u_inc, Cr

      nq_ = task.model.nq
      Qd_l = toB(jp.diag(jp.concatenate([jp.ones(nq_), 0.1 * jp.ones(nx - nq_)]))); R_l = toB(1e-2 * jp.eye(nu)); PN_l = 10.0 * Qd_l
      def cycle_level(jc_):
        LV = [sig_min * 10.0 ** i for i in range(int(round(np.log10(sig0 * 10.0 ** v_up / sig_min))) + 1)]
        s_ref = sgoal(u_inc, sig_min)
        accs = [abs(sgoal(u_inc, l_) - s_ref) / max(r_inc, 1e-12) for l_ in LV]
        if v_level == 'tr':   # trust region in model space: start at sig0, contract x10 per unproductive cycle
          s_ = sig0 * 10.0 ** (v_top - jc_)
          return (s_ if s_ >= sig_min * 0.999 else None), accs
        i_star = max(i_ for i_, a_ in enumerate(accs) if a_ <= kappa or i_ == 0)
        return (LV[i_star + jc_] if i_star + jc_ < len(LV) else None), accs
      def readout_ip(Sip, sig_, P_):
        Wg = toB(Sip.cur.w[:, :nu]); Xg = toB(Sip.cur.x); Cg = toB(Sip.cur.w[:, nu:])
        Ll = S._lin(toL(Xg), toL(Wg), toL(Cg), toL(jp.asarray(S.base * sig_)), toL(P_), toL(jp.zeros((N, nx))), toL(jp.asarray(mu0)))
        K = S._tvlqr(toB(Ll[0]), toB(Ll[1]), Qd_l, R_l, PN_l)
        t0 = time.perf_counter()
        rs, Us_ = true_both(toC(jp.asarray([0.0, 1.0])), QmC, g0C, toC(Xg), toC(Wg), toC(K))
        rs = np.asarray(rs); ib = int(np.argmin(rs)); T['true'] += time.perf_counter() - t0
        return float(rs[ib]), toB(Us_[ib])
      def cycle_start(us, sig_, mode):
        """Start (X, U, C) of a fidelity-cycle excursion at level sig_.  track: TVLQR tracking of the incumbent's TRUE
        trajectory under the smoothed law (an exact smoothed-law trajectory that stays at the incumbent; the mirror of the
        closed-loop read-out); autotrack: the smoothed simulation if it reproduces r* within kappa, else track."""
        us = toB(us)
        if mode == 'lifted':   # the incumbent itself in the lifted variables: true states, controls, TRUE contact forces
          Xt, Ct = truem._rollout(x0C, toC(us), jp.asarray(0.0))
          if bool(jp.all(jp.isfinite(Xt))):
            return toB(Xt), us, toB(Ct[:, :S.ncf])
          mode = 'sim'
        if mode == 'autotrack':
          Xs_, Cs_ = sim(us, sig_)
          if abs(float(plan_goal(QmB, g0B, Xs_)) - r_inc) <= kappa * max(r_inc, 1e-3):
            return Xs_, us, Cs_
          mode = 'track'
        if mode == 'track':
          Xt = toB(truem._rollout(x0C, toC(us), jp.asarray(0.0))[0])
          if bool(jp.all(jp.isfinite(Xt))):
            e_ = toB(jp.asarray(S.base * sig_))
            Ct = S._forces(Xt[:-1], us, e_)
            Ll = S._lin(toL(Xt), toL(us), toL(Ct), toL(e_), toL(p_base), toL(jp.zeros((N, nx))), toL(jp.asarray(mu0)))
            K = S._tvlqr(toB(Ll[0]), toB(Ll[1]), Qd_l, R_l, PN_l)
            Xc, Uc, Cc = S._track(x0B, Xt, us, K, e_)
            if bool(jp.all(jp.isfinite(Xc))):
              return Xc, Uc, Cc
          mode = 'sim'
        Xr_, Cr_ = restart_point(us, sig_, mode)
        return Xr_, us, Cr_
      def explore_local(us, sig_c, sig_f, mu_f, cap, guard_ref):
        """Local fidelity excursion (handoff 4.6 step 1): LFDDP from the incumbent on a MIXED model -- smoothing sig_c and
        the discovery barrier mu0 on the stages where the incumbent's true rollout has no contact force, the fine level
        (sig_f, mu_f) where it has.  Read-outs, ratio test and stall as in the feasible phase; before the true law
        responds only the cap ends it (as the coarse IP excursion).  Updates the incumbent."""
        global SL
        nonlocal r_inc, u_inc, its_ok, t_ok
        t0 = time.perf_counter()
        if SL is None:
          from lfddp_v_perstage import LFDDP as _LV
          SL = _LV(task, lms, vel_w=vel_w, curv=curv, clear_w=opt('--clear-w', None, float), clear_margin=opt('--clear-margin', 0.0))
        Xt_, Ct_ = truem._rollout(x0C, toC(us), jp.asarray(0.0))
        nrm_ = np.asarray(Ct_)[:, :S.ncf] @ np.asarray(S.G).T
        idle = np.all(nrm_ < 1e-6, axis=1)
        if not idle.any():
          return 'busy', 0, 0
        epsv = toB(jp.asarray(S.base * np.where(idle, sig_c, sig_f))); muv = toB(jp.asarray(np.where(idle, mu0, mu_f)))
        P_ = toB(p_base); U_ = toB(us); X_, C_ = SL._roll(x0B, U_, epsv)
        if abs(float(plan_goal(QmB, g0B, X_)) - r_inc) > kappa * max(r_inc, 1e-3):   # 'auto' start, as cycle_start
          Xt_ = toB(Xt_)
          if bool(jp.all(jp.isfinite(Xt_))):
            X_, C_ = Xt_, SL._forces(Xt_[:-1], U_, epsv)
        J_ = float(SL._cost(X_, U_, C_, P_, muv)); reg_ = reg0; lam_ = jp.zeros((N, nx))
        last_resp = 0; lp_ = 0; rp_ = r_inc; k_it = 0; hand = 'cap'; P_rec = R_rec = None; stream = []
        g_ref = r0 if guard_ref is None else guard_ref
        while True:
          k_it += 1
          L_ = SL._lin(toL(X_), toL(U_), toL(C_), toL(epsv), toL(P_), toL(lam_), toL(muv)); L_ = toB(L_)
          fb_, hb_ = L_[8], L_[9]
          G0_ = float(jp.sum(jp.abs(fb_)) + jp.sum(jp.abs(hb_)))
          acc = False
          for _try in range(8):
            k_, K_, d1_, d2_, ok_, lam_n = SL._bwd(L_, U_, toB(jp.asarray(reg_)))
            if not bool(ok_):
              reg_ = max(reg_ * 10, 1e-4); continue
            Xb, Ub, Cb, Jb = SL._fwd_b(X_, U_, C_, k_, K_, fb_, hb_, toB(jp.asarray(ALPHAS)), epsv, x0B, P_, muv, toB(jp.asarray(0.0)))
            Jb = np.asarray(Jb)
            for ia, al in enumerate(ALPHAS):
              ex_ = al * float(d1_) + 0.5 * al * al * float(d2_)
              if G0_ > 1e-9:
                okm = np.isfinite(Jb[ia]) and Jb[ia] + nu_merit * (1 - al) * G0_ < J_ + nu_merit * G0_
              else:
                okm = np.isfinite(Jb[ia]) and Jb[ia] - J_ <= 1e-4 * ex_ and Jb[ia] < J_
              if okm:
                acc = True; break
            if acc:
              break
            reg_ = max(reg_ * 10, 1e-4)
            if reg_ > 1e8:
              break
          if not acc:
            hand = 'converged'; break
          X_, U_, C_, J_ = toB(Xb[ia]), toB(Ub[ia]), toB(Cb[ia]), float(Jb[ia]); lam_ = lam_n
          reg_ = min(max(reg_ * LMF[al], 1e-9), 1e6)
          t1 = time.perf_counter()
          rs, Us_ = true_both(toC(jp.asarray([0.0, 1.0])), QmC, g0C, toC(X_), toC(U_), toC(K_))
          rs = np.asarray(rs); ib = int(np.argmin(rs)); r = float(rs[ib]); T['true'] += time.perf_counter() - t1
          p = float(plan_goal(QmB, g0B, X_))
          mis = False
          if P_rec is None:
            P_rec, R_rec = p, r
          else:
            if P_rec - p > tau * P_rec and r_inc < g_ref and (R_rec - r) < eta * (P_rec - p) and last_resp > 0:
              mis = True
            P_rec, R_rec = min(P_rec, p), min(R_rec, r)
          stream.append((k_it, round(p, 5), round(r, 5)))
          if r < r_inc:
            first_resp = last_resp == 0
            r_inc, u_inc, last_resp = r, toB(Us_[ib]), k_it
            if first_resp or r_inc < (1.0 - tau) * rp_:
              lp_, rp_ = k_it, r_inc
            if its_ok is None and r_inc < 0.01:
              its_ok = ip_its + it + k_it; t_ok = time.perf_counter() - tic
          if r_inc <= stop_tol:
            hand = 'done'; break
          if mis:
            hand = 'mistrust'; break
          if last_resp > 0 and k_it - lp_ >= stall:
            hand = 'stall'; break
          if k_it >= cap:
            hand = 'cap'; break
        T['ip'] += time.perf_counter() - t0
        streams.append(dict(sig=sig_c, r_start=None, stream=stream, hand='local-' + hand, idle=int(idle.sum())))
        return hand, k_it, int(idle.sum())
      def explore(Us, Xs, Cs, sig_, boosts_, cap, stall_=None, guard_ref=None, stall_guard=False, ro_lqr=False, hard_cap=None,
                  hess_=None):
        """Lifted IP exploration (PIP loop: IP chunks of 5, true-law read-outs, ratio test, stall) from (Xs, Us, Cs)
        at smoothing sig_ until its first model-change event.  Updates the incumbent."""
        nonlocal r_inc, u_inc, its_ok, t_ok
        t0 = time.perf_counter()
        with _PLOCK:   # the transcription's parameters are shared by concurrent rounds
          trans.params = Pof(boosts_)
          prm = lms._params(dict(PHip, smoothing=sig_, **({} if hess_ is None else dict(hess=[hess_, 0, 0]))))
          trans.params = p_base
        prm['hom'] = jp.asarray([sig_, 1.0, 0.0, sig_, S.base])
        Sip = HS.start(prm, toG(Xs), toG(jp.concatenate([toB(Us), toB(Cs)], 1)), 5, max_iter=max(cap, hard_cap or 0))
        P_rec = None; R_rec = None; last_acc = 0; r_ref = r_inc; lp_ = 0; rp_ = r_inc
        use_prog = prog_stall >= 1 and stall_guard
        stall_ = ip_stall if stall_ is None else stall_
        g_ref = r0 if guard_ref is None else guard_ref
        stream = []
        while True:
          it_ip = int(Sip.it)
          nxt = None
          if ro_overlap:   # same read-out, but the next IP chunk runs on the GPU while the CPU reads out this iterate
            t1 = time.perf_counter()
            Wg_ = toB(Sip.cur.w[:, :nu]); Xg_ = toB(Sip.cur.x); Cg_ = toB(Sip.cur.w[:, nu:])
            lin_ = lambda: S._lin(toL(Xg_), toL(Wg_), toL(Cg_), toL(jp.asarray(S.base * sig_)), toL(Pof(boosts_)),
                                  toL(jp.zeros((N, nx))), toL(jp.asarray(mu0)))
            if ro_lqr and DL is GPU:   # GPU work of the read-out goes before the chunk on the stream
              Ll = lin_(); AB_ = (toB(Ll[0]), toB(Ll[1])); jax.block_until_ready(AB_)
            WC_, XC_ = toC(Wg_), toC(Xg_); jax.block_until_ready((WC_, XC_, Cg_))
            p = float(plan_goal(toG(Qm), toG(g0), Sip.cur.x))
            if HS.paused(Sip) and it_ip < max(cap, hard_cap or 0):
              nxt = HS.resume(prm, Sip, it_ip + 5)
            T['ro'] += time.perf_counter() - t1
            t1 = time.perf_counter()
            if ro_lqr:
              if DL is not GPU:
                Ll = lin_(); AB_ = (toB(Ll[0]), toB(Ll[1]))
              K = S._tvlqr(*AB_, Qd_l, R_l, PN_l)
              rs, Us_ = true_both(toC(jp.asarray([0.0, 1.0])), QmC, g0C, XC_, WC_, toC(K))
              rs = np.asarray(rs); ib = int(np.argmin(rs)); r, Wg = float(rs[ib]), toB(Us_[ib])
            else:
              r = float(true_eval(QmC, g0C, x0C, WC_)); Wg = Wg_
            T['true'] += time.perf_counter() - t1
          else:
            if ro_lqr:
              r, Wg = readout_ip(Sip, sig_, Pof(boosts_))
            else:
              Wg = Sip.cur.w[:, :nu]; r = tg(Wg)
            p = float(plan_goal(toG(Qm), toG(g0), Sip.cur.x))
          mis = False
          if P_rec is None:
            P_rec, R_rec = p, r
          else:
            if P_rec - p > tau * P_rec and r_inc < g_ref and (R_rec - r) < eta * (P_rec - p):
              mis = True
            P_rec, R_rec = min(P_rec, p), min(R_rec, r)
          stream.append((it_ip, round(p, 5), round(r, 5)))
          if r < r_inc:
            first_resp = last_acc == 0
            r_inc, u_inc, last_acc = r, toB(Wg), it_ip
            if first_resp or r_inc < (1.0 - tau) * rp_:   # the progress window starts at the true law's first response
              lp_, rp_ = it_ip, r_inc
            if its_ok is None and r_inc < 0.01:
              its_ok = it_ip; t_ok = time.perf_counter() - tic
          if r_inc <= stop_tol:
            hand = 'done'; break
          # stall_guard: before the true law has responded in this excursion (no new incumbent yet) only the cap ends it
          stalled = it_ip - (lp_ if use_prog else last_acc) >= stall_ and not (stall_guard and last_acc == 0)
          # hard_cap: the discovery cap only binds before the true law has responded; afterwards stall/mistrust end the
          # excursion (a budget stop is not a stall), bounded by hard_cap (the run's remaining budget)
          capped = it_ip >= cap if (hard_cap is None or last_acc == 0) else it_ip >= hard_cap
          if mis or stalled or capped or not HS.paused(Sip):
            hand = 'mistrust' if mis else ('stall' if stalled else 'cap'); break
          Sip = nxt if nxt is not None else HS.resume(prm, Sip, it_ip + 5)
        T['ip'] += time.perf_counter() - t0
        streams.append(dict(sig=sig_, r_start=r_ref, stream=stream, hand=hand))
        return Sip, hand, it_ip

      # ---------------- phase A: lifted IP exploration ----------------
      if warm == 'init':
        zp = np.load(init_plan)
        X, U, C = toB(jp.asarray(zp['xs'])), toB(jp.asarray(zp['us'])), toB(jp.asarray(zp['cs']))
        r = tg(U)
        if r < r_inc:
          r_inc, u_inc = r, U
        handoff = 'init'
      elif warm == 'fixed':
        t0 = time.perf_counter()
        prm = lms._params(PHip)
        Sip = lms.solver.solve(prm, toG(X), toG(jp.concatenate([jp.zeros((N, nu)), C], 1)), max_iter=warm_ip)
        Xi, Wi = np.asarray(Sip.cur.x), np.asarray(Sip.cur.w)
        ip_its = int(Sip.it); T['ip'] += time.perf_counter() - t0
        Ui = toB(jp.asarray(Wi[:, :nu])); r = tg(Ui)
        if r < r_inc:
          r_inc, u_inc = r, Ui
        X, U, C = toB(jp.asarray(Xi)), Ui, toB(jp.asarray(Wi[:, nu:]))
        handoff = 'fixed'
        if sig_ddp > 0:
          sig = sig_ddp
      elif warm == 'pip' and U_from is not None and lpec_start == 'fine' and lpec_round:
        # LPEC round, fine start: the step is a step on the true law's linearization with a chosen contact branch;
        # continue with the feasible phase at the finest level (the ladder's end state: sig_min, barrier cut at every
        # sharpening, no boost) instead of re-running discovery from a coarse level.  Cycles remain available.
        sig = sig_min; mu = mu0 * mu_fac ** int(round(np.log10(sig0 / sig_min))); boosts = 0
        handoff = 'lpec-fine'; ip_its = 0; Sip = None
        events.append(('lpec-fine', round(r_inc, 4)))
        X, U, C = cycle_start(U, sig, 'auto')
      elif warm == 'pip':
        if U_from is not None:   # the first exploration is a coarse excursion from an incumbent: cycle settings
          s_first = cycle_level(0)[0] if v_level in ('acc', 'tr') else sig0
          events.append(('from-us', round(r_inc, 4), s_first, cycle_level(0)[1]))
          X, U_s, C = cycle_start(U, s_first, v_restart or restart)
          Sip, handoff, ip_its = explore(U_s, X, C, s_first, 0, v_cap if v_cap > 0 else warm_ip, v_stall if v_stall > 0 else None,
                                         None, v_guard >= 2, ip_ro == 'lqr', (total_budget - 10) if v_guard >= 3 else None,
                                         ip_hess_cyc)
          sig = s_first
        else:
          Sip, handoff, ip_its = explore(U, X, C, sig0, 0, warm_ip, ro_lqr=(ip_ro == 'lqr' and ip_ro_scope == 'all'))
        # PIP's model-management decision at the hand-off
        if handoff != 'done':
          if sig_ddp > 0:
            sig = sig_ddp
          elif sig > sig_min * 1.0001 and abs(sgoal(u_inc, sig) - sgoal(u_inc, sig_min)) > kappa * r_inc:
            sig = sig * 0.1; mu = mu * (mu_fac if handoff_mucut else 1.0)
          else:
            boosts = 1
          events.append(('handoff-' + handoff, ip_its, sig, boosts))
        X, U, C = start_point(Sip, sig, handoff_from)
      P = Pof(boosts)
      # ---------------- phase B: LFDDP ----------------
      lam = jp.zeros((N, nx))
      reg = reg0; it = 0; last_acc = 0; trace = []; reason = 'budget'; n_acc = 0; nu_r = 2.0; steps_log = []
      last_prog = 0; r_prog = r_inc
      P_rec = None; R_rec = None; alpha = None; last_change = 0; n_cycle = 0; n_v = 0
      J = float(S._cost(X, U, C, toB(P), toB(jp.asarray(mu))))
      while it < budget + v_budget * n_v and handoff != 'done' and r_inc > stop_tol and ip_its + it < total_budget + v_budget * n_v:
        it += 1
        eps = jp.asarray(S.base * sig)
        t0 = time.perf_counter()
        L = S._lin(toL(X), toL(U), toL(C), toL(eps), toL(P), toL(lam), toL(jp.asarray(mu))); L = jax.block_until_ready(L)
        T['lin'] += time.perf_counter() - t0
        t0 = time.perf_counter(); L = toB(L); T['xfer'] += time.perf_counter() - t0
        fbar, hbar = L[8], L[9]
        G0 = float(jp.sum(jp.abs(fbar)) + jp.sum(jp.abs(hbar)))
        Jtask = J - float(jp.sum(jax.vmap(S.bar, (0, None))(C, toB(jp.asarray(mu)))))
        accepted = False; conv = False
        muB, epsB, PB = toB(jp.asarray(mu)), toB(eps), toB(P); ksB = toB(jp.asarray(kscale))
        if ls_spec and ls_mode == 'par' and lm != 2:
          # speculative attempts (same result as the sequential loop below): a rejected attempt is always followed by
          # one at reg' = max(10 reg, 1e-4), so that backward pass is computed up front and both line searches run at
          # once (CPU devices 0-4 and 5-9); they are evaluated in the sequential order.
          def _ls(bw, off):
            k_, K_ = bw[0], bw[1]
            return [S._fwd(*[jax.device_put(t_, CPUS[off + ia]) for t_ in (X, U, C, k_, K_, fbar, hbar)],
                           jax.device_put(jp.asarray(al), CPUS[off + ia]),
                           *[jax.device_put(t_, CPUS[off + ia]) for t_ in (epsB, x0B, PB, muB, ksB)])
                    for ia, al in enumerate(ALPHAS)]
          def _bw(rg_):
            t0 = time.perf_counter()
            o = S._bwd(L, U, toB(jp.asarray(rg_))); T['n_bwd'] += 1
            o = (o[0], o[1], float(o[2]), float(o[3]), bool(o[4]), o[5]); T['bwd'] += time.perf_counter() - t0
            return o
          def _accept(outs, bw):
            for ia, al in enumerate(ALPHAS):
              Jn_ = float(outs[ia][3]); ex_ = al * bw[2] + 0.5 * al * al * bw[3]
              if G0 > 1e-9:
                okm_ = np.isfinite(Jn_) and Jn_ + nu_merit * (1 - al) * G0 < J + nu_merit * G0
              else:
                okm_ = np.isfinite(Jn_) and Jn_ - J <= 1e-4 * ex_ and Jn_ < J
              if okm_:
                return ia, al, Jn_
            return None
          n_try = 0; pend = None
          while n_try < 8:
            bwA = pend if pend is not None else _bw(reg)
            pend = None
            if not bwA[4]:
              reg = max(reg * 10, 1e-4); n_try += 1; continue
            if G0 <= 1e-9 and abs(bwA[2]) < tol_d * max(Jtask, 1e-3):
              conv = True; break
            reg_n = max(reg * 10, 1e-4)
            bwB = _bw(reg_n) if (n_try + 1 < 8 and reg_n <= 1e8) else None   # before the line searches occupy the CPUs
            t0 = time.perf_counter()
            outA = _ls(bwA, 0); T['n_ls'] += 1
            outB = _ls(bwB, 5) if (bwB is not None and bwB[4] and not (G0 <= 1e-9 and abs(bwB[2]) < tol_d * max(Jtask, 1e-3))) else None
            resA = _accept(outA, bwA)
            if resA is not None:
              ia, alpha, Jn = resA; k, K, d1, d2, ok, lam_n = bwA
              Xn, Un, Cn = toB(outA[ia][0]), toB(outA[ia][1]), toB(outA[ia][2]); accepted = True
              T['fwd'] += time.perf_counter() - t0; break
            T['fwd'] += time.perf_counter() - t0
            reg = reg_n; n_try += 1
            if reg > 1e8 or bwB is None:
              break
            if outB is None:      # attempt B: Cholesky failure or converged -> as the sequential loop
              pend = bwB; continue
            T['n_ls'] += 1; t0 = time.perf_counter()
            resB = _accept(outB, bwB)
            T['fwd'] += time.perf_counter() - t0
            if resB is not None:
              ia, alpha, Jn = resB; k, K, d1, d2, ok, lam_n = bwB
              Xn, Un, Cn = toB(outB[ia][0]), toB(outB[ia][1]), toB(outB[ia][2]); accepted = True; break
            reg = max(reg * 10, 1e-4); n_try += 1
            if reg > 1e8:
              break
        else:
          for _try in range(8):
            t0 = time.perf_counter()
            k, K, d1, d2, ok, lam_n = S._bwd(L, U, toB(jp.asarray(reg)))
            T['n_bwd'] += 1
            d1, d2, ok = float(d1), float(d2), bool(ok)
            T['bwd'] += time.perf_counter() - t0
            if not ok:
              if lm == 2:
                reg = max(reg * nu_r, 1e-4); nu_r *= 2.0
              else:
                reg = max(reg * 10, 1e-4)
              continue
            if verbose:
              print(f'  it {it} try {_try} reg {reg:.1e} d1 {d1:.3e} J {J:.4f} G0 {G0:.2e}', flush=True)
            if G0 <= 1e-9 and abs(d1) < tol_d * max(Jtask, 1e-3):
              conv = True; break
            t0 = time.perf_counter()
            T['n_ls'] += 1
            if ls_mode == 'par':   # one step size per CPU device, dispatched concurrently
              outs = []
              for ia, al in enumerate(ALPHAS):
                dv = CPUS[ia]
                outs.append(S._fwd(*[jax.device_put(t_, dv) for t_ in (X, U, C, k, K, fbar, hbar)],
                                   jax.device_put(jp.asarray(al), dv), *[jax.device_put(t_, dv) for t_ in (epsB, x0B, PB, muB, ksB)]))
              Xb = [o[0] for o in outs]; Ub = [o[1] for o in outs]; Cb = [o[2] for o in outs]
              Jb = np.array([float(o[3]) for o in outs])
            elif ls_mode == 'batch':
              Xb, Ub, Cb, Jb = S._fwd_b(X, U, C, k, K, fbar, hbar, toB(jp.asarray(ALPHAS)), epsB, x0B, PB, muB, ksB)
              Jb = np.asarray(Jb)
            for ia, al in enumerate(ALPHAS):
              if ls_mode == 'seq':
                Xn, Un, Cn, Jn = S._fwd(X, U, C, k, K, fbar, hbar, toB(jp.asarray(al)), epsB, x0B, PB, muB, ksB)
                Jn = float(Jn)
              else:
                Jn = float(Jb[ia])
              exp = al * d1 + 0.5 * al * al * d2
              if G0 > 1e-9:
                okm = np.isfinite(Jn) and Jn + nu_merit * (1 - al) * G0 < J + nu_merit * G0
              else:
                okm = np.isfinite(Jn) and Jn - J <= 1e-4 * exp and Jn < J
              if okm:
                if ls_mode != 'seq':
                  Xn, Un, Cn = toB(Xb[ia]), toB(Ub[ia]), toB(Cb[ia])
                accepted = True; alpha = al; break
            T['fwd'] += time.perf_counter() - t0
            if accepted:
              break
            if lm == 2:
              reg = max(reg * nu_r, 1e-4); nu_r *= 2.0
            else:
              reg = max(reg * 10, 1e-4)
            if reg > 1e8:
              break
        mis = False
        if accepted:
          J_prev = J
          X, U, C, J = Xn, Un, Cn, Jn; lam = lam_n
          if lm == 2:   # gain-ratio rule: actual over predicted decrease of the acceptance test's merit
            pred_ = -(alpha * d1 + 0.5 * alpha * alpha * d2)
            if G0 > 1e-9:
              rho_ = ((J_prev + nu_merit * G0) - (Jn + nu_merit * (1 - alpha) * G0)) / max(pred_ + nu_merit * alpha * G0, 1e-300)
            else:
              rho_ = (J_prev - Jn) / max(pred_, 1e-300)
            reg = min(max(reg * max(1.0 / 3.0, 1.0 - (2.0 * rho_ - 1.0) ** 3), 1e-9), 1e6); nu_r = 2.0
            steps_log.append((alpha, round(float(rho_), 4), float(reg)))
          else:
            reg = min(max(reg * LMF[alpha], 1e-9), 1e6) if lm else max(reg / 3.0, 1e-9)
            if log_steps:
              steps_log.append((alpha, None, float(reg)))
          n_acc += 1
          if n_acc % ro_every == 0:
            t0 = time.perf_counter()
            if readout == 'fb':
              sv = np.array([0.0, 1.0])
              if par_ro:   # open-loop and feedback read-outs on two CPU devices, concurrently
                o2 = [true_one(*[jax.device_put(t_, CPUS[5 + i]) for t_ in (jp.asarray(sv[i]), QmC, g0C, X, U, K)]) for i in range(2)]
                rs = np.array([float(o[0]) for o in o2]); Us = [o[1] for o in o2]
              else:
                rs, Us = true_both(toC(jp.asarray(sv)), QmC, g0C, toC(X), toC(U), toC(K))
              rs = np.asarray(rs); ib = int(np.argmin(rs)); r = float(rs[ib]); cand = toB(Us[ib])
            else:
              r = float(true_eval(QmC, g0C, x0C, toC(U))); cand = U
            T['true'] += time.perf_counter() - t0
            if r < r_inc:
              r_inc, u_inc, last_acc = r, toB(cand), it
              if r_inc < (1.0 - tau) * r_prog:
                last_prog, r_prog = it, r_inc
              if its_ok is None and r_inc < 0.01:
                its_ok = ip_its + it; t_ok = time.perf_counter() - tic
            if mistrust:   # ratio test: the (feasible, smoothed-law) plan improves, the true law does not follow
              p = float(plan_goal(QmB, g0B, X))
              if P_rec is None:
                P_rec, R_rec = p, r
              else:
                if P_rec - p > tau * P_rec and r_inc < r0 and (R_rec - r) < eta * (P_rec - p) and it - last_change >= dwell:
                  mis = True
                P_rec, R_rec = min(P_rec, p), min(R_rec, r)
            trace.append((it, sig, boosts, round(J, 5), round(r, 5), round(r_inc, 5), alpha, reg))
        else:
          conv = True
        if r_inc <= stop_tol:
          reason = 'done'; break
        prog_on = prog_stall == 2 or (prog_stall == 1 and n_v > 0)
        # the feasible phase's budget ends short of the goal: as exhausted as a finished ladder -> fidelity cycle
        force_cycle = (vcyc and v_bud_cycle and n_v < vcyc and warm == 'pip' and r_inc >= 0.01 and it >= budget + v_budget * n_v
                       and ip_its + it < total_budget + v_budget * (n_v + 1))
        if conv or mis or force_cycle or it - (last_prog if prog_on else last_acc) >= stall:
          why = 'mistrust' if mis else ('converged' if conv else 'stall')
          inacc = sig > sig_min * 1.0001 and abs(sgoal(u_inc, sig) - sgoal(u_inc, sig_min)) > kappa * r_inc
          if not force_cycle and sig > sig_min * 1.0001 and (inacc or boosts >= max_boosts):
            sig = max(sig_min, sig * 0.1); mu = mu * mu_fac; events.append((why + '-sharpen', it, sig))
          elif boosts < max_boosts and not force_cycle:
            boosts += 1; P = Pof(boosts); events.append((why + '-boost', it, boosts))
          elif not force_cycle and cycle and n_cycle < cycle and warm == 'pip' and ip_its + it < total_budget:
            # the feasible phase is stuck at a local minimum short of the goal: re-explore with the lifted IP
            n_cycle += 1
            Xr, Cr = restart_point(u_inc, sig, restart)
            Sip, hand, its_c = explore(u_inc, Xr, Cr, sig, boosts, ip_cycle, ro_lqr=(ip_ro == 'lqr' and ip_ro_scope == 'all'))
            ip_its += its_c; events.append(('cycle-' + hand, it, its_c))
            if hand == 'done':
              reason = 'done'; break
            X, U, C = start_point(Sip, sig, handoff_from)
            J = float(S._cost(X, U, C, toB(P), toB(jp.asarray(mu)))); reg = reg0
            last_acc = it; lam = jp.zeros((N, nx)); P_rec = None; last_change = it; last_prog, r_prog = it, r_inc
            continue
          elif vcyc and n_v < vcyc and warm == 'pip' and r_inc >= 0.01 and ip_its + it < total_budget + v_budget * (n_v + 1):
            # fidelity cycle: the fine model is exhausted short of the goal -> re-explore from the incumbent at a coarse
            # level with the discovery barrier; coarsen further (x10) after a cycle without sufficient decrease
            if r_cyc is not None:
              jc = 0 if r_inc < (1.0 - tau) * r_cyc else jc + 1
            sig_c, accs = cycle_level(jc)
            if v_level in ('acc', 'tr'):
              if sig_c is None:
                reason = 'exhausted'; break
            else:
              if jc > v_up:
                reason = 'exhausted'; break
              sig_c = sig0 * 10.0 ** jc
            n_v += 1; r_cyc = r_inc
            if v_local:   # coarse only on the incumbent's idle stages (LFDDP on the mixed model)
              hand, its_c, n_idle = explore_local(u_inc, sig_c, sig, mu, v_cap if v_cap > 0 else warm_ip, r_cyc)
              it += its_c; Sip = None
              events.append(('local-idle', n_idle))
            else:
              Xr, Ur, Cr = cycle_start(u_inc, sig_c, v_restart or restart)
              Sip, hand, its_c = explore(Ur, Xr, Cr, sig_c, 0, v_cap if v_cap > 0 else warm_ip,
                                         v_stall if v_stall > 0 else None, r_cyc if v_guard else None, v_guard >= 2, ip_ro == 'lqr',
                                         min(total_budget - 10, total_budget + v_budget * n_v - ip_its - it) if v_guard >= 3 else None,
                                         ip_hess_cyc)
              ip_its += its_c
            prod = r_inc < r_cyc or v_reset == 'always'
            cyc_log.append(dict(cyc=n_v, it=it, sig=sig_c, its=its_c, hand=hand, start=r_cyc, after_ip=r_inc, jc=jc,
                                accs=[round(a_, 3) for a_ in accs]))
            events.append(('vcycle-' + hand, it, its_c, sig_c, round(r_cyc, 4), round(r_inc, 4)))
            if hand == 'done' or r_inc <= stop_tol:
              reason = 'done'; break
            if prod:   # restart the model ladder at the coarse level (PIP's hand-off decision, as after the first exploration)
              sig, mu, boosts = (sig, mu, 0) if v_local else (sig_c, mu0, 0)   # local: the fine stages never left the fine level
              if sig > sig_min * 1.0001 and abs(sgoal(u_inc, sig) - sgoal(u_inc, sig_min)) > kappa * r_inc:
                sig = sig * 0.1
              else:
                boosts = 1
              P = Pof(boosts)
              X, U, C = start_point(Sip, sig, handoff_from)
            else:      # nothing found: keep the fine model, next cycle one level coarser
              jc += 1; r_cyc = None
              if v_level not in ('acc', 'tr') and jc > v_up:
                reason = 'exhausted'; break
              X, U, C = start_point(None, sig, handoff_from)
              # force the next exhaustion check immediately
            J = float(S._cost(X, U, C, toB(P), toB(jp.asarray(mu)))); reg = reg0
            last_acc = it if prod else it - stall; lam = jp.zeros((N, nx)); P_rec = None; last_change = it
            last_prog, r_prog = last_acc, r_inc
            continue
          else:
            reason = 'exhausted'; break
          U = u_inc; X, C = restart_point(U, sig, restart); J = float(S._cost(X, U, C, toB(P), toB(jp.asarray(mu)))); reg = reg0
          last_acc = it; lam = jp.zeros((N, nx)); P_rec = None; last_change = it; last_prog, r_prog = it, r_inc
      if handoff == 'done' or r_inc <= stop_tol:
        reason = 'done'
      return dict(r=r_inc, u=u_inc, reason=reason, handoff=handoff, sig=sig, mu=mu, boosts=boosts, P=P, events=events,
                  trace=trace, steps=steps_log, cycles=n_v, fine_cycles=n_cycle, cyc_log=cyc_log, streams=streams, its_ok=its_ok, t_ok=t_ok,
                  it=it, ip_its=ip_its)

    best = run_round(U_from, False); tot_it, tot_ip = best['it'], best['ip_its']
    # branch-step fallback: every smoothed level is exhausted short of the goal.  The best plan of all rounds is the
    # incumbent (monotone); a round is accepted if it lowers the best true goal by the fraction tau.  With --lpec-par k,
    # the next k trust radii of the shrink sequence (D, D/2.5, D/6.25) are tried at once, as concurrent rounds.
    while lpec and best['r'] >= 0.01 and len(lp_log) < lpec:
      if lp_step is None:
        from lpec_step import make_lpec
        _grb_step = make_lpec(task, S, truem, lms)
        if lpec_solver == 'fast':   # Gurobi MILP chooses the branches; the structured (parallel-in-time) QP gives the step
          from lpec_fast import make_lpec_fast
          _fast = make_lpec_fast(task, S, truem, lms)
          def lp_step(u_, D_):
            br = _grb_step(u_, D_, branch_only=True)
            if br['nbi'] == 0 or br.get('zs') is None:
              return br
            z_ = np.zeros((S.N, S.G.shape[0]))
            for b_, (k_, j_) in enumerate(br['bi']):
              z_[k_, j_] = br['zs'][b_]
            rf = _fast(u_, D_, mode='milp', milp_z=z_)
            if rf['ok']:
              rf['time'] += br['time']; rf['msg'] = 'fast ' + rf['msg']
              return rf
            rg_ = _grb_step(u_, D_); rg_['msg'] = 'fallback (IP not converged) ' + str(rg_.get('msg'))
            return rg_
        else:
          lp_step = _grb_step
      k_par = max(1, min(lpec_par, 3 - lp_shrink, lpec - len(lp_log)))
      Ds = [lp_D / 2.5 ** i_ for i_ in range(k_par)]
      t_rnd = time.perf_counter(); steps = []; stop = None
      for D_ in Ds:
        stp = lp_step(np.asarray(best['u']), D_)
        ent = dict(delta=D_, nbi=stp['nbi'], t_lpec=round(stp['time'], 1))
        if stp['nbi'] == 0:
          ent['stop'] = 'B-stationary'; lp_log.append(ent); stop = True; break
        if stp.get('U1') is None:
          ent['stop'] = 'no step: ' + stp.get('msg', ''); lp_log.append(ent); stop = True; break
        ent.update(pressed=stp['pressed'], step_goal=round(stp['goal_true'], 4))
        steps.append((ent, stp['U1']))
        print(f"  LPEC round {len(lp_log) + len(steps)}: delta {D_:g}, {stp['nbi']} biactive, pressed {stp['pressed']}, step goal "
              f"{stp['goal_true']:.4f} ({stp['time']:.0f} s); best {best['r']:.4f}", flush=True)
      if not steps:
        break
      if len(steps) == 1:
        outs = [run_round(steps[0][1], True)]
      else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(len(steps)) as ex_:
          outs = list(ex_.map(lambda su: run_round(su[1], True), steps))
      t_b = time.perf_counter() - t_rnd; r_ref = best['r']
      for (ent, _), o in zip(steps, outs):
        tot_it += o['it']; tot_ip += o['ip_its']
        ent.update(goal=round(o['r'], 4), t_round=round(t_b, 1), accepted=bool(o['r'] < (1.0 - tau) * r_ref), parallel=len(steps))
        lp_log.append(ent)
      ob = min(outs, key=lambda o: o['r'])
      if ob['r'] < best['r']:
        best = ob
      acc = [i_ for i_, (ent, _) in enumerate(steps) if ent['accepted']]
      if acc:     # sequential rule: an accepted step keeps its radius and resets the shrink count
        lp_D = Ds[acc[0]]; lp_shrink = 0
      else:
        lp_shrink += len(steps); lp_D = Ds[-1] / 2.5
        if lp_shrink > 2:
          lp_log[-1]['stop'] = 'no improvement'; break
      if stop:
        break
    r_inc, u_inc = best['r'], best['u']
    reason, handoff, sig, mu, boosts, P = (best[k_] for k_ in ('reason', 'handoff', 'sig', 'mu', 'boosts', 'P'))
    events, trace, n_v, n_cycle, cyc_log, streams = (best[k_] for k_ in ('events', 'trace', 'cycles', 'fine_cycles', 'cyc_log', 'streams'))
    its_ok, t_ok = best['its_ok'], best['t_ok']
    it, ip_its = tot_it, tot_ip
    t = time.perf_counter() - tic
    # end-of-run diagnostics of the returned plan (not timed): is a miss an optimization stop or a model mismatch?
    Pd = toB(P); mud = toB(jp.asarray(mu))
    Xd_min, Cd_min = sim(u_inc, sig_min)
    Xt = np.asarray(truem._rollout(x0C, toC(u_inc), jp.asarray(0.0))[0])
    diag = dict(sig=sig, mu=mu, boosts=boosts,
                plan_goal_sigmin=float(plan_goal(QmB, g0B, Xd_min)), plan_goal_sig=sgoal(u_inc, sig),
                term_cost=float(trans.terminal_cost(jp.asarray(np.asarray(Xd_min)[-1]), p_base)),
                barrier=float(jp.sum(jax.vmap(S.bar, (0, None))(Cd_min, mud))),
                true_vs_plan_state_gap=float(np.abs(Xt[-1] - np.asarray(Xd_min)[-1]).max()))
    if os.environ.get('SAVE_DIR'):
      os.makedirs(os.environ['SAVE_DIR'], exist_ok=True)
      np.savez(f"{os.environ['SAVE_DIR']}/{label}_{name}_{seed}.npz", us=np.asarray(u_inc))
    row = dict(task=name, label=label, seed=seed, true_goal=r_inc, true_ok=bool(r_inc < 0.01), it=it, ip_its=ip_its,
               its_ok=its_ok, t_ok=t_ok, reason=reason, handoff=handoff, time=t, diag=diag,
               timing={k_: round(v, 3) for k_, v in T.items()}, events=events, trace=trace, cfg=cfg,
               cycles=n_v, fine_cycles=n_cycle, cyc_log=cyc_log, streams=streams, lpec=lp_log, steps=best.get('steps'))
    print(json.dumps({k_: v for k_, v in row.items() if k_ not in ('trace', 'cfg', 'streams')}), flush=True)
    with open(out, 'a') as fo:
      fo.write(json.dumps(row) + '\n')
