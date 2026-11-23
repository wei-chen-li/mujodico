"""Scalar impulse, logarithmic derivative, and self-consistent step checks."""
import jax
import jax.numpy as jp
import numpy as np
from run import ROOT,patch,dump
from mujodico import mjx,trajopt
from mujodico.examples import tasks
from mujodico.mjx._src import solver_sap_util as su
from mujodico.mjx._src import solver_sap
from mujodico.trajopt._src.transcriptions import _put_optimization_model,_extract_contact_indices

def main():
  result={};task=tasks.PushSphere2d();reference_dynamics=None;reference_jacobian=None
  task.model.opt.tolerance=task.model.opt.ls_tolerance=1e-12
  task.model.opt.iterations=task.model.opt.ls_iterations=100
  model=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=2e-5)
  d=mjx.make_data(model);d=mjx.fwd_position(model,d.replace(qpos=jp.asarray(task.x_0[:model.nq])))
  addresses,_=_extract_contact_indices(model);i=int(addresses[0])
  dt=float(model.opt.timestep);K=float(d._impl.efc_k[i])*solver_sap._STIFFNESS_SCALE
  D=float(d._impl.efc_b[i])*solver_sap._HUNT_CROSSLEY_DAMPING_SCALE
  bounce=solver_sap._HUNT_CROSSLEY_BOUNCE_SPEED_TOL
  result['parameters']={'dt':dt,'stiffness':K,'damping':D,'smoothing':2e-5}
  for label,a,b,g in [('a2',2,2,2),('a1',1,1,1),('a1_width2',1,2,1),('a1_wronggrad',1,2,2)]:
    patch(a,b,g)
    def impulse(v,phi):
      return su.hunt_crossley_normal_impulse(v,phi,dt,K,D,2e-5,bounce)
    def supplied(v,phi):
      return su.hunt_crossley_normal_impulse_grad(v,phi,dt,K,D,2e-5,bounce)
    cases=[]
    for phi,v in [(.38231715287178925,0.),(.03,0.),(.003,-.1),(.01,-.2),(0.,-.1),(-.001,0.)]:
      val=float(impulse(v,phi));dv=float(jax.grad(impulse,0)(v,phi));dp=float(jax.grad(impulse,1)(v,phi))
      cases.append({'gap':phi,'velocity':v,'force':val/dt,'dlogF_dgap':dp/val,
        'dlogF_dvelocity':dv/val,'manual_dimpulse_dvelocity':float(supplied(v,phi)),
        'AD_dimpulse_dvelocity':dv})
    dyn,eq,_,bounds=trajopt.lifted_shooting_transcription(task.model,task.contact_approx,2e-5,task.u_bounds)
    ncf=len(bounds[0])-task.nu
    probe_w=jp.r_[jp.zeros(task.nu),jp.ones(ncf)*.01]
    dv=np.asarray(jax.jit(dyn)(task.x_0,probe_w))
    dj=np.asarray(jax.jit(jax.jacfwd(dyn,0))(task.x_0,probe_w))
    if reference_dynamics is None:reference_dynamics,reference_jacobian=dv,dj
    np.testing.assert_array_equal(dv,reference_dynamics)
    np.testing.assert_array_equal(dj,reference_jacobian)
    def force(y):return -eq(y[:task.nx],jp.r_[y[task.nx:],jp.zeros(ncf)])
    f=jax.jit(force);jac=jax.jit(jax.jacfwd(force))
    n=(task.x_0[2:4]-task.x_0[:2]);n/=np.linalg.norm(n)
    states=[]
    for gap,speed in [(.38231715287178925,0.),(.003,.1),(.01,.2),(.03,.8)]:
      x=task.x_0.copy();x[:2]=x[2:4]-(.13+gap)*n;x[4:6]=n*speed
      u=n*speed;y=jp.r_[x,u];J=np.asarray(jac(y));h=1e-7
      pert=jp.eye(len(y))*h
      fd=np.asarray((jax.jit(jax.vmap(force))(y+pert)-jax.jit(jax.vmap(force))(y-pert))/(2*h)).T
      w=jp.r_[u,f(y)]
      xn=np.asarray(jax.jit(dyn)(x,w))
      actualgap=np.linalg.norm(xn[:2]-xn[2:4])-.13
      states.append({'gap':gap,'pusher_speed':speed,'force':np.asarray(f(y)),
        'jacobian_relative_error':float(np.linalg.norm(J-fd)/np.linalg.norm(fd)),
        'log_force_gradient':J/np.asarray(f(y))[:,None],
        'actual_next_gap':actualgap})
    curve=None
    if label in ['a1','a2']:
      gaps=np.linspace(.0001,.06,121);xx=np.tile(task.x_0,(len(gaps),1))
      xx[:,:2]=xx[:,2:4]-(.13+gaps[:,None])*n;xx[:,4:6]=n*.8
      uu=np.tile(n*.8,(len(gaps),1))
      yy=jp.asarray(np.hstack([xx,uu]))
      ff=np.asarray(jax.jit(jax.vmap(force))(yy))
      ww=jp.asarray(np.hstack([uu,ff]))
      xn=np.asarray(jax.jit(jax.vmap(dyn))(xx,ww))
      curve={'gaps':gaps,'forces':ff[:,0],'next_gaps':np.linalg.norm(xn[:,:2]-xn[:,2:4],axis=1)-.13}
    result[label]={'scalar_cases':cases,'self_consistent_cases':states,'force_curve':curve,
      'given_force_dynamics_difference':float(abs(dv-reference_dynamics).max()),
      'given_force_dynamics_jacobian_difference':float(abs(dj-reference_jacobian).max())}
  dump(ROOT/'probes.json',result)
  print(result['parameters'],flush=True)

if __name__=='__main__':main()
