"""Decompose saved IPOPT KKT residuals and check contact Jacobian accuracy."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import numpy as np
from scipy import sparse

from mujodico import trajopt, mjx
from mujodico.examples import methods, tasks
from mujodico.mjx._src import solver_sap
from mujodico.trajopt._src.transcriptions import _extract_contact_indices, _put_optimization_model

jax.config.update('jax_compilation_cache_dir', '/tmp/mujodico-contact-jax-cache')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'convergence' / 'data'


def dump(path, value):
  def convert(v):
    if isinstance(v, bytes): return v.decode()
    if isinstance(v, (jax.Array, np.ndarray)): return np.asarray(v).tolist()
    if isinstance(v, np.generic): return v.item()
    if hasattr(v, 'value'): return v.value
    raise TypeError(type(v).__name__)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, default=convert, indent=2))


def build(label, exact=False, inner_tolerance=None, inner_iterations=None):
  saved = json.loads((ROOT / 'data' / (label+'.json')).read_text())
  a = saved['args']
  task = getattr(tasks, a['task'])(a['seed'])
  if a.get('condim') is not None: task.model.geom_condim[:] = a['condim']
  if a.get('elliptic'):
    import mujoco
    task.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  tol = inner_tolerance if inner_tolerance is not None else a.get('inner_tolerance')
  nit = inner_iterations if inner_iterations is not None else a.get('inner_iterations')
  if tol is not None:
    task.model.opt.tolerance = tol
    task.model.opt.ls_tolerance = tol
  if nit is not None:
    task.model.opt.iterations = nit
    task.model.opt.ls_iterations = nit
  object.__setattr__(task, 'contact_smoothing', a['smoothing']*1000)
  trans = methods.LiftedMultipleShootingTranscription(task)
  opt = trajopt.IpoptTrajOpt(
      N=trans.N, nx=trans.nx, nu=trans.nw,
      local_cost=lambda x,w,xn,k,p: trans.stage_cost(x,w,p),
      terminal_cost=trans.terminal_cost,
      initial_equality_constraint=lambda x,p: x-p.x_0,
      local_equality_constraint=lambda x,w,xn,k,p: jp.concatenate([
          xn-trans.dynamics(x,w), trans.stage_equality(x,w)]),
      local_inequality_constraint=lambda x,w,xn,k,p: trans.stage_inequality(x,w),
      x_bounds=trans.x_bounds, u_bounds=trans.w_bounds,
      params_trace=trans.params, gauss_newton=not exact,
  )
  raw = np.load(ROOT / 'data' / (label+'.npz'))
  return saved, task, trans, opt._nlp, raw


def state_names(task):
  if task.model.nq == 3: return ['pusher_x','pusher_y','knob_angle','pusher_vx','pusher_vy','knob_omega']
  if task.model.nq == 4: return ['pusher_x','pusher_y','object_x','object_y','pusher_vx','pusher_vy','object_vx','object_vy']
  return ['pusher_x','pusher_y','object_x','object_y','object_angle','pusher_vx','pusher_vy','object_vx','object_vy','object_omega']


def describe_variable(index, task, trans):
  nx, nw, N = trans.nx, trans.nw, trans.N
  stage, local = divmod(int(index), nx+nw)
  if stage == N or local < nx:
    return {'stage':stage,'kind':'state','component':state_names(task)[local], 'index':int(index)}
  local -= nx
  return {'stage':stage,'kind':'control' if local < task.nu else 'force',
          'component':['pusher_ux','pusher_uy'][local] if local < task.nu else f'contact_{local-task.nu}',
          'index':int(index)}


def analyze(label):
  saved, task, trans, nlp, raw = build(label)
  xs, ws = raw['xs'], raw['ws']
  z = nlp._pack(xs,ws)
  c = np.asarray(nlp.constraints(z))
  ncf = trans.nw-task.nu
  local = c[trans.nx:].reshape(trans.N,-1)
  dyn, force = local[:,:trans.nx], local[:,trans.nx:trans.nx+ncf]
  info = saved['info']
  if 'mult_g' not in info: raise ValueError('Saved IPOPT multipliers required')
  lam = np.asarray(info['mult_g'])
  zl, zu = np.asarray(info['mult_x_L']),np.asarray(info['mult_x_U'])
  grad = np.asarray(nlp.gradient(z))
  # The public structure is a list of pairs, not a tuple of coordinate arrays.
  ij = np.asarray(nlp.jacobian_structure())
  J = sparse.coo_matrix((nlp.jacobian_entries(z),(ij[:,0],ij[:,1])),shape=(nlp.num_constraints,nlp.num_vars)).tocsr()
  rd = grad+J.T@lam-zl+zu
  lb, ub = nlp.var_bounds
  ml, mu = np.isfinite(lb),np.isfinite(ub)
  delta_l = np.minimum(1e-4,1e-8*np.maximum(1,np.abs(lb[ml])))
  delta_u = np.minimum(1e-4,1e-8*np.maximum(1,np.abs(ub[mu])))
  sl, su = z[ml]-(lb[ml]-delta_l),(ub[mu]+delta_u)-z[mu]
  cp_l, cp_u = sl*zl[ml],su*zu[mu]
  hist = json.loads((ROOT/'data'/(label+'_history.json')).read_text())
  target = hist[-1]['mu']
  cp = np.concatenate([cp_l,cp_u])
  top_dual = []
  for index in np.argsort(-np.abs(rd))[:12]:
    d = describe_variable(index,task,trans)
    d.update(residual=rd[index],objective_gradient=grad[index],
             equality_gradient=(J.T@lam)[index],lower_multiplier=zl[index],upper_multiplier=zu[index])
    d['primal_value']=z[index]
    top_dual.append(d)
  bound_complementarity=[]
  for side,indices,products,slacks,mults in [('lower',np.flatnonzero(ml),cp_l,sl,zl[ml]),
                                           ('upper',np.flatnonzero(mu),cp_u,su,zu[mu])]:
    for j in np.argsort(-np.abs(products))[:6]:
      d=describe_variable(indices[j],task,trans)
      d.update(side=side,product=products[j],slack=slacks[j],multiplier=mults[j],
               residual_from_mu=products[j]-target)
      bound_complementarity.append(d)
  force_order = np.argsort(-np.abs(force).ravel())[:12]
  model = _put_optimization_model(task.model,contact_approximation=task.contact_approx,
                                 contact_smoothing=saved['args']['smoothing'])
  data = mjx.make_data(model)
  addresses,_ = _extract_contact_indices(model)
  @jax.jit
  def contact_info(x,u):
    d = data.replace(qpos=x[:model.nq],qvel=x[model.nq:model.nq+model.nv],ctrl=u)
    d = mjx.forward(model,d)
    vn = d._impl.efc_J[addresses] @ (d.qvel+model.opt.timestep*d.qacc)
    K = d._impl.efc_k[addresses]*solver_sap._STIFFNESS_SCALE
    hc_d = d._impl.efc_b[addresses]*solver_sap._HUNT_CROSSLEY_DAMPING_SCALE
    eps = jp.sqrt(8*saved['args']['smoothing']/K)
    x1 = -d._impl.efc_pos[addresses]-2*model.opt.timestep*vn
    return d._impl.efc_force[addresses],d._impl.contact.dist,vn,K,hc_d,eps,x1-eps/2
  all_contact = [np.asarray(v) for v in jax.jit(jax.vmap(contact_info))(
      jp.asarray(xs[:-1]),jp.asarray(ws[:,:task.nu]))]
  top_force = []
  for index in force_order:
    k, edge = np.unravel_index(index,force.shape)
    d = {'stage':int(k),'time_seconds':float(k*task.model.opt.timestep),
         'edge':int(edge),'residual':force[k,edge], 'lifted_force':ws[k,task.nu+edge],
         'forward_force':all_contact[0][k,edge], 'gap':all_contact[1][k,0],
         'next_edge_velocity':all_contact[2][k,edge], 'stiffness':all_contact[3][k,edge],
         'damping':all_contact[4][k,edge], 'epsilon':all_contact[5][k,edge],
         'penetration_branch_margin':all_contact[6][k,edge]}
    top_force.append(d)
  worst_dyn = np.unravel_index(np.argmax(np.abs(dyn)),dyn.shape)
  # One-stage contact equality finite differences at the largest violation.
  k = top_force[0]['stage']
  f = lambda v: trans.stage_equality(v[:trans.nx],v[trans.nx:])
  v = jp.asarray(np.concatenate([xs[k],ws[k]]))
  analytic = np.asarray(jax.jit(jax.jacfwd(f))(v))
  finite_differences = []
  for h in [1e-4,1e-6,1e-8]:
    eye = jp.eye(len(v))*h
    fd = np.asarray(jax.jit(jax.vmap(lambda dv:(f(v+dv)-f(v-dv))/(2*h)))(eye)).T
    finite_differences.append({'step':h,'max_absolute_error':float(np.max(np.abs(fd-analytic))),
       'max_relative_row_error':float(np.max(np.linalg.norm(fd-analytic,axis=1)/np.maximum(1,np.linalg.norm(analytic,axis=1)))),
       'jacobian':fd})
  result = {
      'label':label,'task':saved['args']['task'],'last_iteration':hist[-1],
      'initial_state_defect_max':float(np.max(np.abs(c[:trans.nx]))),
      'dynamics_defect_max':float(np.max(np.abs(dyn))),
      'dynamics_defect_per_component':dict(zip(state_names(task),np.max(np.abs(dyn),axis=0))),
      'worst_dynamics_stage':int(worst_dyn[0]),'worst_dynamics_component':state_names(task)[worst_dyn[1]],
      'force_equality_defect_max':float(np.max(np.abs(force))),
      'force_rows_exceeding_1e_minus4':int(np.sum(np.abs(force)>1e-4)),
      'bounds_violation_max':saved['bound_violation_max'],
      'stationarity_max_unscaled':float(np.max(np.abs(rd))),
      'complementarity_max_relaxed_bounds':float(np.max(np.abs(cp))),
      'perturbed_complementarity_max':float(np.max(np.abs(cp-target))),
      'complementarity_min':float(np.min(cp)),'barrier_mu':target,
      'multiplier_max':float(max(np.max(np.abs(lam)),np.max(zl),np.max(zu))),
      'top_stationarity':top_dual,'top_force_defects':top_force,
      'top_bound_complementarity':sorted(bound_complementarity,key=lambda x:-abs(x['product'])),
      'force_equality_jacobian_at_worst_stage':analytic,
      'finite_differences_at_worst_stage':finite_differences,
      'last100':{key:{'min':min(h[key] for h in hist[-100:]),'max':max(h[key] for h in hist[-100:])}
                 for key in ['objective','inf_pr','inf_du','mu','alpha_pr','alpha_du','d_norm']},
  }
  dump(OUT/(label+'_kkt.json'),result)
  np.savez_compressed(OUT/(label+'_residuals.npz'),dynamics=dyn,force=force,
                       stationarity=rd,complementarity_lower=cp_l,complementarity_upper=cp_u,
                       lower_bound_indices=np.flatnonzero(ml),upper_bound_indices=np.flatnonzero(mu))
  print(label,{k:result[k] for k in ['dynamics_defect_max','force_equality_defect_max',
        'stationarity_max_unscaled','complementarity_max_relaxed_bounds','perturbed_complementarity_max','barrier_mu']},flush=True)


if __name__ == '__main__':
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--labels',nargs='+',default=['sphere_baseline','box_baseline','circular_baseline','square_baseline'])
  args=ap.parse_args()
  for label in args.labels: analyze(label)
