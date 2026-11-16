"""Check lateral gap and generalized-force derivatives at the initial states."""
import json
import jax
import jax.numpy as jp
import numpy as np
from trace import ROOT,build,encode
from mujodico import mjx
from mujodico.examples import tasks
from mujodico.trajopt._src.transcriptions import _extract_contact_indices

def main():
  result={}
  for name in ['PushSphere2d','PushBox2d']:
    task=getattr(tasks,name)();tr,m,d0,_=build(task,'baseline')
    addr,_=_extract_contact_indices(m)
    def geometry(x):
      d=mjx.fwd_position(m,d0.replace(qpos=x[:m.nq],qvel=x[m.nq:]))
      return d._impl.contact.dist[0],d._impl.efc_J[addr[0]]
    x=jp.asarray(task.x_0);gap,J=jax.jit(geometry)(x)
    n=J[2:4];t=jp.array([-n[1],n[0]])
    delta=jp.zeros(task.nx).at[:2].set(t)
    _,(dg,dJ)=jax.jit(lambda x,v:jax.jvp(geometry,(x,),(v,)))(x,delta)
    eps=1e-6
    fp=jax.jit(geometry)(x+eps*delta);fm=jax.jit(geometry)(x-eps*delta)
    fd=(fp[1]-fm[1])/(2*eps)
    np.testing.assert_allclose(dJ,fd,atol=1e-8,rtol=1e-6)
    if name=='PushSphere2d':
      expected=-t/np.linalg.norm(x[:2]-x[2:4])
      np.testing.assert_allclose(dJ[2:4],expected,atol=1e-10)
    else:
      # Here t is minus the box's local positive x direction, so dτ/dt=-λ.
      np.testing.assert_allclose(dJ[2:4],0.,atol=1e-10)
      np.testing.assert_allclose(dJ[4],-1.,atol=1e-10)
    result[name]={'gap':gap,'normal':n,'tangent':t,
      'lateral_gap_derivative':dg,'lateral_generalized_force_derivative_per_newton':dJ,
      'finite_difference_max_error':float(abs(dJ-fd).max())}
  (ROOT/'geometric_probes.json').write_text(json.dumps(result,default=encode,indent=2))
  print(json.dumps(result,default=encode,indent=2))

if __name__=='__main__':main()
