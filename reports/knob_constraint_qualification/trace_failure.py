"""Log rank, Newton accuracy, and filter decisions for the failed IPPAS case."""
import json
from types import SimpleNamespace
import jax
import jax.numpy as jp
import numpy as np
from run import ROOT,DATA,build,dump
from mujodico import trajopt
from mujodico.trajopt._src.ippas import ip_search_dir,ip_linesearch
from mujodico.trajopt._src.ippas import ip_primal_dual_system as pd
from mujodico.trajopt._src.ippas.ip_iterate import max_abs
from mujodico.trajopt._src.ippas.ip_iterate import distances,damping

def main():
  cfg=SimpleNamespace(**json.loads((DATA/'two_copied_ippas.json').read_text())['args'])
  task,tr,_,_,_,nw,dyn,eq,bounds=build(cfg)
  records=[]
  def record(phase,values):
    row={'phase':phase,**{k:float(v) for k,v in values.items()}}
    records.append(row);print(json.dumps(row),flush=True)
  compute=ip_search_dir.compute
  def wrapped_compute(system,mu,last_delta,options):
    direction,good,delta=compute(system,mu,last_delta,options)
    sv=jp.linalg.svd(system.Eu,compute_uv=False)
    jax.debug.callback(lambda v:record('newton',v),{
      'minimum_stage_equality_control_singular_value':sv.min(),
      'maximum_stage_equality_control_singular_value':sv.max(),
      'linear_system_defect':max_abs(*pd.system_defect(direction).values()),
      'direction_valid':good,'primal_regularization':delta,
      'dual_regularization':ip_search_dir.dual_regularization(system.Eu,mu,options),
      'max_force_step':abs(direction.du[:,task.nu:]).max(),
      'max_joint_position_step':abs(direction.dx[:,:task.nu]).max()},ordered=True)
    return direction,good,delta
  search=ip_linesearch.find_acceptable_trial_point
  def wrapped_search(model,it,data,direction,mu,params,bounds,state,options):
    found,newstate,rejected=search(model,it,data,direction,mu,params,bounds,state,options)
    p0,t0=ip_linesearch.merit(it,vars(data),mu,bounds,options)
    p1,t1=ip_linesearch.merit(found.iterate,found.evaluation,mu,bounds,options)
    amax,_=ip_linesearch.maximum_step_lengths(it,direction,mu,bounds,options)
    dobj=jp.sum(data.objective_gradx*direction.dx)+jp.sum(data.objective_gradu*direction.du)
    dbarrier=jp.asarray(0.)
    for s,ds,l,u in [(it.s,direction.ds,*bounds[:2]),(it.sf,direction.dsf,*bounds[2:])]:
      dl,du=distances(s,l,u)
      dbarrier+=jp.sum(ds*(-mu*jp.where(jp.isfinite(l),1/dl,0.)
                         +mu*jp.where(jp.isfinite(u),1/du,0.)+damping(l,u,mu,options)))
    jax.debug.callback(lambda v:record('line_search',v),{
      'accepted':found.accepted,'maximum_fraction_to_boundary_step':amax,'last_trial_alpha':found.alpha,
      'barrier_objective_before':p0,'barrier_objective_last_trial':p1,
      'violation_l1_before':t0,'violation_l1_last_trial':t1,
      'task_objective_directional_derivative':dobj,'slack_barrier_directional_derivative':dbarrier,
      'any_filter_rejected':rejected},ordered=True)
    return found,newstate,rejected
  ip_search_dir.compute=wrapped_compute
  ip_linesearch.find_acceptable_trial_point=wrapped_search
  opt=trajopt.IppasTrajOpt(N=tr.N,nx=tr.nx,nu=nw,
      initial_state=lambda p:p.x_0,dynamics=lambda x,w,k,p:dyn(x,w),
      stage_cost=lambda x,w,k,p:tr.stage_cost(x,w,p),terminal_cost=tr.terminal_cost,
      stage_equality_constraint=lambda x,w,k,p:eq(x,w),x_bounds=tr.x_bounds,u_bounds=bounds,
      params_trace=tr.params)
  w0=np.hstack([task.us_init,np.zeros((tr.N,nw-task.nu))])
  out=jax.jit(lambda x,w:opt.solve(tr.params,x,w,max_iter=500))(task.xs_init,w0)
  jax.block_until_ready(out)
  dump(ROOT/'failed_ippas_newton_trace.json',records)

if __name__=='__main__':main()
