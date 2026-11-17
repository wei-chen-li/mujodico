"""Distinguish structural rank, zero-force degeneracy, and physical authority."""
import argparse
import json
from types import SimpleNamespace
import jax
import jax.numpy as jp
import numpy as np
from run import ROOT,DATA,build,dump
from mujodico import mjx
from mujodico.trajopt._src.transcriptions import _extract_contact_indices

def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--label',required=True)
  args=ap.parse_args();saved=json.loads((DATA/(args.label+'.json')).read_text())
  cfg=SimpleNamespace(**saved['args']);task,tr,solver,m,d,nw,dyn,eq,bounds=build(cfg)
  raw=np.load(DATA/(args.label+'.npz'));xs=raw['xs'];ws=raw['ws']
  if cfg.form!='lifted':raise ValueError('Force-equality diagnostics require lifting')
  ncf=nw-task.nu;addresses,contacts=_extract_contact_indices(m)
  def force(x,u):return -tr.stage_equality(x,jp.r_[u,jp.zeros(ncf)])
  predict=jax.jit(jax.vmap(force));derivative=jax.jit(jax.vmap(jax.jacfwd(force,(0,1))))
  forces=np.asarray(predict(xs[:-1],ws[:,:task.nu]))
  Fx,Fu=map(np.asarray,derivative(xs[:-1],ws[:,:task.nu]))
  gradnorm=np.sqrt((Fx*Fx).sum(2)+(Fu*Fu).sum(2));zeros=(forces==0)
  def geometry(x):
    nd=mjx.fwd_position(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:]))
    return nd._impl.efc_J[addresses],nd._impl.contact.dist,nd._impl.contact.geom
  J,gap,pairs=map(np.asarray,jax.jit(jax.vmap(geometry))(xs));pairs=pairs[0]
  # Rank of sticking velocity constraints for the distal contacts only. The
  # unused out-of-plane tangent is omitted; it is a planar mechanism.
  if cfg.task in ['finger','two']:
    tips=[i for i,p in enumerate(pairs) if any(int(g) in ([1] if cfg.task=='finger' else [1,3]) for g in p)]
  else:tips=[0]
  normal=[];tangent=[]
  for i in tips:
    Ji=J[:,4*i:4*i+4,:]
    normal.append(Ji.mean(1))
    diffs=[(Ji[:,0]-Ji[:,1])/2,(Ji[:,2]-Ji[:,3])/2]
    tangent.append(max(diffs,key=lambda a:np.linalg.norm(a)))
  C=np.stack([a for pair in zip(normal,tangent) for a in pair],axis=1)
  sv=np.linalg.svd(C,compute_uv=False)
  j0=J[0].reshape(-1,4,m.nv)
  duplicate_pairs=[(i,int(np.linalg.norm(Ji[0]-Ji[1])==0),int(np.linalg.norm(Ji[2]-Ji[3])==0)) for i,Ji in enumerate(j0)]
  @jax.jit
  def fullstep_info(x,w):
    nd=mjx.forward(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:],ctrl=w[:task.nu]))
    step=mjx.step(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:],ctrl=w[:task.nu]))
    f=nd._impl.efc_force[addresses]
    reduced=tr.dynamics(x,jp.r_[w[:task.nu],f])
    full=jp.r_[step.qpos,step.qvel,step.act]
    mask=np.ones(d._impl.nefc,bool);mask[addresses]=False
    return full-reduced,nd._impl.efc_force[mask]
  difference,other_forces=map(np.asarray,jax.jit(jax.vmap(fullstep_info))(xs[:-1],ws))
  # Exactly feasible zero-control and brief retreat trajectories: eliminate
  # forces, then integrate using the same given-force dynamics as the NLP.
  @jax.jit
  def rollout(us):
    def advance(x,u):
      f=force(x,u);xn=tr.dynamics(x,jp.r_[u,f]);return xn,(xn,f)
    _,(tail,fs)=jax.lax.scan(advance,jp.asarray(task.x_0),us)
    return jp.concatenate([task.x_0[None],tail]),fs
  probes={}
  for name,pulse in [('cold',0.),('retreat',1.)]:
    us=np.zeros((tr.N,task.nu))
    if cfg.task in ['finger','two']:us[:25,0]=pulse
    else:us[:25,:]=-.2*pulse
    xx,ff=map(np.asarray,rollout(us))
    ww=np.hstack([us,ff])
    dynamics_defect=float(abs(xx[1:]-np.asarray(jax.jit(jax.vmap(tr.dynamics))(xx[:-1],ww))).max())
    force_defect=float(abs(np.asarray(jax.jit(jax.vmap(tr.stage_equality))(xx[:-1],ww))).max())
    dx,du=map(np.asarray,derivative(xx[:-1],us))
    fnorm=np.sqrt((dx*dx).sum(2)+(du*du).sum(2))
    zz=(ff==0)
    bounds_violation=float(np.maximum(np.maximum(task.x_bounds[0]-xx,xx-task.x_bounds[1]),0).max())
    probes[name]={'force_zero_count':int(zz.sum()),'force_zero_with_zero_gradient':int((zz&(fnorm==0)).sum()),
      'minimum_force':float(ff.min()),'state_bound_violation':bounds_violation,
      'dynamics_defect':dynamics_defect,'force_equality_defect':force_defect,
      'minimum_finite_state_bound_slack':float(np.min(np.r_[
        (xx-task.x_bounds[0])[:,np.isfinite(task.x_bounds[0])].ravel(),
        (task.x_bounds[1]-xx)[:,np.isfinite(task.x_bounds[1])].ravel()])),
      'angle_degrees':float(np.rad2deg(xx[-1,m.nq-1]))}
  result={'label':args.label,'original_solver_violation':saved['violation'],
    'force_equalities':ncf*tr.N,'dynamics_equalities':tr.nx*(tr.N+1),
    'control_degrees_of_freedom':task.nu*tr.N,'equality_full_row_rank_by_triangular_identity_blocks':True,
    'actual_force_zeros':int(zeros.sum()),'actual_force_zeros_with_zero_gradient':int((zeros&(gradnorm==0)).sum()),
    'minimum_positive_force':float(forces[forces>0].min(initial=np.inf)),
    'near_bound_force_decisions_1e-8':int((ws[:,task.nu:]<=1e-8).sum()),
    'initial_distal_contact_kinematic_singular_values':sv[0],
    'minimum_distal_contact_kinematic_singular_value':float(sv.min()),
    'initial_duplicate_pyramid_row_pairs':duplicate_pairs,
    'max_given_contact_force_vs_full_step_difference':float(abs(difference).max()),
    'max_noncontact_constraint_force':float(abs(other_forces).max(initial=0)),
    'nodes_with_noncontact_force_over_1e-6':int((abs(other_forces).max(1,initial=0)>1e-6).sum()),
    'state_bounds_within_1e-6':int((np.minimum(xs-task.x_bounds[0],task.x_bounds[1]-xs)<1e-6).sum()),
    'feasible_trajectory_probes':probes}
  if cfg.task in ['finger','two']:
    result['proximal_minimum_possible_gap_m']=float(np.hypot(.15,.25)-.15-.03-.08)
    result['contact_gap_minimums']=gap.min(0)
  dump(ROOT/(args.label+'_diagnostics.json'),result)
  np.savez_compressed(ROOT/(args.label+'_diagnostics.npz'),forces=forces,force_gradient_norm=gradnorm,
       kinematic_singular_values=sv,forward_discrepancy=difference,other_forces=other_forces)
  print(json.dumps(result,default=lambda x:np.asarray(x).tolist(),indent=2))

if __name__=='__main__':main()
