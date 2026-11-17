"""Kinematic sticking-contact continuation until a joint/workspace limit.

This probes a single contact mode, not maximum physically achievable rotation.
Force feasibility, actuator effort, and the time budget are not imposed here.
"""
import json
from types import SimpleNamespace
import jax
import jax.numpy as jp
import numpy as np
from scipy.integrate import solve_ivp
from run import ROOT,DATA,build,dump
from mujodico import mjx
from mujodico.trajopt._src.transcriptions import _extract_contact_indices

def main():
  out={}
  for label in ['quadrant','finger','two']:
    saved=json.loads((DATA/(label+'.json')).read_text());cfg=SimpleNamespace(**saved['args'])
    task,tr,_,m,d,_,_,_,_=build(cfg)
    raw=np.load(DATA/(label+'.npz'));xs=raw['smooth_xs']
    addresses,_=_extract_contact_indices(m)
    tips=[0] if label=='quadrant' else ([1] if label=='finger' else [1,3])
    @jax.jit
    def contact(q):
      nd=mjx.fwd_position(m,d.replace(qpos=q))
      J=nd._impl.efc_J[addresses].reshape(-1,4,m.nv)
      rows=[]
      for i in tips:
        ji=J[i];normal=ji.mean(0)
        t1=(ji[0]-ji[1])/2;t2=(ji[2]-ji[3])/2
        tangent=jp.where(jp.linalg.norm(t1)>jp.linalg.norm(t2),t1,t2)
        rows.extend([normal,tangent])
      return jp.stack(rows),nd._impl.contact.dist[jp.array(tips)]
    _,gaps=jax.jit(jax.vmap(contact))(xs[:,:m.nq])
    mask=np.all(abs(np.asarray(gaps))<.0002,axis=1)
    lower,upper=task.x_bounds[0][:m.nq],task.x_bounds[1][:m.nq]
    mask&=np.all((xs[:,:m.nq]>=lower)&(xs[:,:m.nq]<=upper),axis=1)
    candidates=np.flatnonzero(mask)
    if not len(candidates):out[label]={'error':'No simultaneous distal-contact replay node.'};continue
    k=int(candidates[0]);q0=xs[k,:m.nq]
    @jax.jit
    def rate(q):
      C,_=contact(q)
      return jp.r_[jp.linalg.solve(C[:,:-1],-C[:,-1]),1.]
    @jax.jit
    def joint_sv(q):return jp.linalg.svd(contact(q)[0][:,:-1],compute_uv=False)[-1]
    def limit(t,q):return float(np.minimum(q-lower,upper-q).min())
    def singular(t,q):return float(joint_sv(q))-1e-5
    limit.terminal=singular.terminal=True
    limit.direction=singular.direction=-1
    solution=solve_ivp(lambda t,q:np.asarray(rate(q)),(0,7.),q0,events=[limit,singular],
                       rtol=1e-8,atol=1e-10,max_step=.005)
    qs=solution.y.T;reached=qs[-1]
    _,gg=jax.jit(jax.vmap(contact))(qs)
    C,_=contact(q0)
    Cend,_=contact(reached)
    _,_,VT=np.linalg.svd(Cend,full_matrices=True)
    result={'start_replay_node':k,'start_configuration':q0,'start_gap':gaps[k],
      'sticking_velocity_jacobian_singular_values':np.linalg.svd(C,compute_uv=False),
      'end_full_contact_constraint_singular_values':np.linalg.svd(Cend,compute_uv=False),
      'end_nullspace_knob_component':float(abs(VT[-1,-1])),
      'start_relative_contact_velocities':np.asarray(C)@xs[k,m.nq:],
      'additional_rotation_before_limit_degrees':float(np.rad2deg(reached[-1]-q0[-1])),
      'end_configuration':reached,'integrator_status':solution.status,
      'termination':('joint/workspace bound' if len(solution.t_events[0]) else
                     'joint-to-contact kinematic singularity' if len(solution.t_events[1]) else solution.message),
      'end_joint_contact_singular_value':float(joint_sv(reached)),
      'maximum_gap_drift_m':float(abs(np.asarray(gg)-np.asarray(gaps[k])).max()),
      'unit_knob_speed_joint_velocities':np.asarray(rate(q0))[:-1]}
    out[label]=result
    np.savez_compressed(ROOT/(label+'_sticking_stroke.npz'),q=qs)
  dump(ROOT/'sticking_strokes.json',out)
  print(json.dumps(out,default=lambda a:np.asarray(a).tolist(),indent=2))

if __name__=='__main__':main()
