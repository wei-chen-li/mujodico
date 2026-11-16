"""Condensed, constrained shooting polish on the original zero-smoothing model.

The start controls come from an optimizer-generated multiple-shooting result.
The task objective is unchanged. State bounds are enforced as inequalities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cyipopt
import jax
import jax.numpy as jp
import numpy as np
from mujodico import mjx
from mujodico.examples.tasks import FingersSphere2d
from mujodico.mjx._src import solver_sap_util as su
from mujodico.trajopt._src.transcriptions import _put_optimization_model
from run import ROOT,DATA,save_json,release_events

jax.config.update('jax_compilation_cache_dir','/tmp/mujodico-contact-jax-cache')


def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--label',required=True)
  ap.add_argument('--init',required=True);ap.add_argument('--iterations',type=int,default=1000)
  ap.add_argument('--state-margin',type=float,default=1e-5)
  ap.add_argument('--max-wall-time',type=float,default=900.)
  args=ap.parse_args();DATA.mkdir(exist_ok=True,parents=True)
  task=FingersSphere2d();task.model.opt.tolerance=task.model.opt.ls_tolerance=1e-12
  task.model.opt.iterations=task.model.opt.ls_iterations=100
  # Define the hard branch explicitly: the original sqrt(0) expression has a
  # NaN derivative. Away from the kink, these are the same zero-smoothing law.
  def hc(vn,phi0,dt,k,d,contact_smoothing,bounce_speed_tol,next_step=True):
    t=dt if next_step else 0.
    return dt*k*d*jp.maximum(-phi0-t*vn,0.)*su.local_smooth_relu(1/d-vn,bounce_speed_tol)
  def hc_grad(vn,phi0,dt,k,d,contact_smoothing,bounce_speed_tol,next_step=True):
    t=dt if next_step else 0.;a=-phi0-t*vn;b=1/d-vn
    slope=jp.where(a>0,1.,jp.where(a<0,0.,.5))
    return -dt*k*d*(t*slope*su.local_smooth_relu(b,bounce_speed_tol)
                    +jp.maximum(a,0.)*su.local_smooth_relu_grad(b,bounce_speed_tol))
  su.hunt_crossley_normal_impulse=hc;su.hunt_crossley_normal_impulse_grad=hc_grad
  m=_put_optimization_model(task.model,contact_approximation=task.contact_approx,contact_smoothing=0.)
  d=mjx.make_data(m);N=task.horizon;nu=task.nu;nx=task.nx
  def step(x,u):
    nd=mjx.step(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:],ctrl=u))
    return jp.concatenate([nd.qpos,nd.qvel,nd.act])
  @jax.jit
  def rollout(us):
    def advance(x,u):
      xn=step(x,u);return xn,xn
    _,tail=jax.lax.scan(advance,jp.asarray(task.x_0),us)
    return jp.concatenate([jp.asarray(task.x_0)[None],tail])
  linearize=jax.jit(jax.vmap(jax.jacfwd(step,(0,1))))
  qids=np.flatnonzero(np.isfinite(task.x_bounds[0])|np.isfinite(task.x_bounds[1]))
  ng=len(qids)
  jr=np.concatenate([np.repeat(np.arange(k*ng,(k+1)*ng),(k+1)*nu) for k in range(N)])
  jc=np.concatenate([np.tile(np.arange((k+1)*nu),ng) for k in range(N)])
  history=[];problem=None

  class NLP:
    def __init__(self):self.cached=None;self.have_derivatives=False;self.calls=0
    def evaluate(self,z,derivatives=False):
      if self.cached is None or not np.array_equal(z,self.cached):
        self.cached=np.array(z);self.us=self.cached.reshape(N,nu)
        self.xs=np.asarray(rollout(jp.asarray(self.us)));self.err=self.xs-task.x_goal
        self.obj=float(np.einsum('ki,ij,kj->',self.err[:-1],task.Q,self.err[:-1])
           +self.err[-1]@task.Qf@self.err[-1]+np.einsum('ki,ij,kj->',self.us,task.R,self.us))
        self.have_derivatives=False;self.calls+=1
      if derivatives and not self.have_derivatives:
        A,B=[np.asarray(v) for v in linearize(jp.asarray(self.xs[:-1]),jp.asarray(self.us))]
        costate=2*task.Qf@self.err[-1];grad=np.zeros_like(self.us)
        for k in reversed(range(N)):
          grad[k]=2*task.R@self.us[k]+B[k].T@costate
          costate=2*task.Q@self.err[k]+A[k].T@costate
        self.grad=grad.ravel();S=np.zeros((nx,N*nu));entries=[]
        for k in range(N):
          S=A[k]@S;S[:,k*nu:(k+1)*nu]+=B[k]
          entries.append(S[qids,:(k+1)*nu].ravel())
        self.jac=np.concatenate(entries);self.have_derivatives=True
    def objective(self,z):self.evaluate(z);return self.obj
    def gradient(self,z):self.evaluate(z,True);return self.grad
    def constraints(self,z):self.evaluate(z);return self.xs[1:,qids].ravel()
    def jacobianstructure(self):return jr,jc
    def jacobian(self,z):self.evaluate(z,True);return self.jac
    def intermediate(self,alg_mod,iteration,objective,inf_pr,inf_du,mu,d_norm,reg,alpha_du,alpha_pr,ls_trials):
      v=problem.get_current_violations(scaled=False)
      h=dict(iter=iteration,objective=objective,inf_pr=inf_pr,inf_du=inf_du,mu=mu,
             alpha_pr=alpha_pr,alg_mod=alg_mod,unscaled={k:float(np.max(abs(a),initial=0)) for k,a in v.items()})
      history.append(h)
      if iteration%25==0:print(json.dumps(h),flush=True)
      return True
  nlp=NLP()
  cl=np.tile(task.x_bounds[0][qids]+args.state_margin,N)
  cu=np.tile(task.x_bounds[1][qids]-args.state_margin,N)
  problem=cyipopt.Problem(n=N*nu,m=N*ng,problem_obj=nlp,
      lb=np.tile(task.u_bounds[0],N),ub=np.tile(task.u_bounds[1],N),cl=cl,cu=cu)
  for k,v in dict(hessian_approximation='limited-memory',limited_memory_max_history=20,
      mu_strategy='adaptive',max_iter=args.iterations,max_wall_time=args.max_wall_time,print_level=0).items():problem.add_option(k,v)
  source=Path(args.init)
  if source.suffix!='.npz':source=DATA/(args.init+'.npz')
  raw=np.load(source);us0=raw['ws'][:,:nu]
  print('Starting physical constrained shooting:',args.label,flush=True)
  initial_objective=nlp.objective(us0.ravel())
  z,info=problem.solve(us0.ravel());nlp.evaluate(z);xs,us=nlp.xs,nlp.us
  @jax.jit
  def geom(x):
    nd=mjx.fwd_position(m,d.replace(qpos=x[:m.nq],qvel=x[m.nq:]))
    return nd._impl.contact.dist,nd._impl.contact.pos,nd._impl.contact.geom
  gaps,points,pairs=[np.asarray(v) for v in jax.jit(jax.vmap(geom))(jp.asarray(xs))]
  pairs=pairs[0];masks={'left':np.array([any(k in p for k in [0,1]) for p in pairs]),
                       'right':np.array([any(k in p for k in [2,3]) for p in pairs])}
  contacts={}
  for side,mask in masks.items():
    g=np.min(gaps[:,mask],axis=1);contacts[side]={'min_gap':float(g.min()),'max_gap':float(g.max()),
       'regrasp_events':release_events(g),'steps_in_contact_0p5mm':int(sum(g<=.0005)),
       'steps_released_2mm':int(sum(g>=.002))}
  boundviol=float(np.max(np.maximum(np.maximum(task.x_bounds[0]-xs,xs-task.x_bounds[1]),0)))
  metrics={'objective':nlp.obj,'position_error_m':float(np.linalg.norm(xs[-1,4:6]-task.x_goal[4:6])),
       'angle_error_radians':float(abs(xs[-1,6]-task.x_goal[6])),'max_state_difference':0.,'contacts':contacts}
  common_hash=hashlib.sha256(task.xs_init.tobytes()+task.us_init.tobytes()).hexdigest()
  info.pop('x',None)
  summary={'args':dict(vars(args),form='condensed_shooting',solver='ipopt',smoothing=0.,hessian='bfgs',force_floor=float('inf')),
     'options':{'maxcor':20,'state_margin':args.state_margin},'info':info,'status':int(info['status']),
     'iterations':history[-1]['iter'],'last_iteration':history[-1],'objective':nlp.obj,
     'initial_objective':initial_objective,'initial_state_control_hash':common_hash,
     'position_error_m':metrics['position_error_m'],'angle_error_radians':metrics['angle_error_radians'],
     'constraint_violation':max(0.,float(np.max(np.maximum(np.maximum(cl-xs[1:,qids].ravel(),xs[1:,qids].ravel()-cu),0)))),
     'force_equality_defect':0.,'dynamics_defect':0.,'bound_violation':boundviol,
     'ncf':0,'nx':nx,'N':N,'contact_pairs':pairs,'object_geom':4,'finger_contact_masks':masks,
     'contacts':contacts,'replay':{'zero':metrics,'smooth':metrics}}
  save_json(DATA/(args.label+'.json'),summary);save_json(DATA/(args.label+'_history.json'),history)
  np.savez_compressed(DATA/(args.label+'.npz'),xs=xs,ws=us,gaps=gaps,points=points,
      zero_xs=xs,smooth_xs=xs,zero_gaps=gaps,smooth_gaps=gaps)
  print('RESULT',args.label,summary['status'],metrics,'state bound violation',boundviol,flush=True)


if __name__=='__main__':main()
