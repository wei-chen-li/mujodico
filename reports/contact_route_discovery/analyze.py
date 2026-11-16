"""Summarize traced routes and resolve geometric Lagrangian derivatives."""
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trace import DATA,ROOT,build,encode
from mujodico.examples import tasks
from mujodico import mjx
from mujodico.trajopt._src.transcriptions import _extract_contact_indices

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)

def unpack(z,nx,nw):
  a=z[:-nx].reshape(-1,nx+nw)
  return np.vstack([a[:,:nx],z[-nx:]]),a[:,nx:]

def main():
  summaries={};routes={}
  for f in DATA.glob('*.json'):
    s=json.loads(f.read_text());r=np.load(f.with_suffix('.npz'));nx,nw=s['nx'],s['nw']
    rows=[]
    for i,z in enumerate(r['iterates']):
      x,w=unpack(z,nx,nw);v=x[:,:2]-x[:,2:4];a=np.unwrap(np.arctan2(v[:,1],v[:,0]))
      end_angle=float(np.degrees(a[-1]-a[0]))
      rows.append({'iteration':s['history'][i]['iter'],'span':float(np.degrees(np.ptp(a))),
        'net_angle':end_angle,'error':float(np.linalg.norm(x[-1,2:4])),
        'force_max':float(w[:,2:].max()),'pT':x[-1,:2],'oT':x[-1,2:4],
        'thetaT':float(x[-1,4]) if nx==10 else None})
    routes[f.stem]=rows
    summaries[f.stem]={k:s[k] for k in ['position_error','angle_span_degrees','max_primal_value_change_from_ablation']}
    summaries[f.stem].update(status=s['info']['status'],net_angle=rows[-1]['net_angle'],
        first_90deg_iteration=next((a['iteration'] for a in rows if a['span']>90),None))
  (ROOT/'routes.json').write_text(json.dumps(routes,default=encode,indent=2))
  (ROOT/'summary.json').write_text(json.dumps(summaries,default=encode,indent=2))
  fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
  for i,base in enumerate(['sphere','box']):
    for key in [base,base+'_no_geometry']:
      if key in routes:ax[i].plot([r['iteration'] for r in routes[key]],[r['span'] for r in routes[key]],label=key)
    ax[i].set(xlabel='Interior-point iteration',ylabel='Pusher/object angle span (degrees)',title=base)
    ax[i].legend();ax[i].grid(alpha=.2)
  fig.savefig(FIG/'geometry_ablation.png',dpi=180);plt.close(fig)

  for label,iterations in [('sphere',[2,20,70,77,90,150]),('box',[2,4,5,8,10,70])]:
    r=np.load(DATA/(label+'.npz'));s=json.loads((DATA/(label+'.json')).read_text())
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,it in zip(axs.flat,iterations):
      index=next(i for i,h in enumerate(s['history']) if h['iter']==it)
      x,w=unpack(r['iterates'][index],s['nx'],s['nw'])
      ax.plot(x[:,:2][:,0],x[:,:2][:,1],color='tab:red',label='Pusher trajectory')
      ax.plot(x[:,2],x[:,3],'--',color='tab:blue',label='Object trajectory')
      ax.scatter(*x[0,:2],color='tab:red',marker='s',s=15)
      ax.scatter(*x[-1,:2],color='tab:red',s=20)
      ax.scatter(0,0,marker='x',color='k',label='Object goal')
      if label=='sphere':
        ax.add_patch(plt.Circle(x[0,2:4],.08,color='tab:blue',alpha=.15))
      else:
        theta=x[0,4];c,ss=np.cos(theta),np.sin(theta)
        box=np.array([[-.15,-.08],[.15,-.08],[.15,.08],[-.15,.08]])@np.array([[c,ss],[-ss,c]])+x[0,2:4]
        ax.add_patch(plt.Polygon(box,color='tab:blue',alpha=.15))
      ax.set(title=f'Optimizer iteration {it}',aspect='equal',xlim=(-.23,.62),
             ylim=(-.53,.4) if label=='box' else (-.23,.45))
      ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=7)
    fig.savefig(FIG/(label+'_iterates.png'),dpi=180);plt.close(fig)

  fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
  for key in ['symmetric','offset_plus','offset_minus']:
    if key not in routes:continue
    axs[0].plot([r['iteration'] for r in routes[key]],[r['net_angle'] for r in routes[key]],label=key)
    rr=np.load(DATA/(key+'.npz'));x=rr['xs']
    axs[1].plot(x[:,0],x[:,1],label=key)
  axs[0].set(xlabel='Interior-point iteration',ylabel='Net change in relative bearing (degrees)')
  axs[1].set(xlabel='Pusher x (m)',ylabel='Pusher y (m)',aspect='equal')
  for ax in axs:ax.legend();ax.grid(alpha=.2)
  fig.savefig(FIG/'symmetry.png',dpi=180);plt.close(fig)

  all_details={}
  for label in ['sphere','box']:
    path=DATA/(label+'.npz')
    if not path.exists():continue
    r=np.load(path);s=json.loads(path.with_suffix('.json').read_text())
    task=getattr(tasks,s['args']['task'])();tr,m,d0,_=build(task,'baseline')
    addresses,_=_extract_contact_indices(m)
    @jax.jit
    def geometry(x):
      d=mjx.fwd_position(m,d0.replace(qpos=x[:m.nq],qvel=x[m.nq:]))
      return d._impl.efc_J[addresses[0],2:4],d._impl.contact.dist[0]
    def geometry_effect(x,w):
      return tr.dynamics(x,w)-tr.dynamics(x,w.at[task.nu:].set(0.))
    linear=jax.jit(jax.vmap(jax.jacfwd(geometry_effect,0)))
    cf_linear=jax.jit(jax.vmap(jax.jacfwd(tr.stage_equality,0)))
    geom=jax.jit(jax.vmap(geometry))
    selected=list(range(min(len(r['iterates'])-1,15)))+[20,30,40,50,60,70,80,90]
    details=[]
    for index in selected:
      if index>=len(r['iterates'])-1:continue
      xs,ws=unpack(r['iterates'][index],tr.nx,tr.nw)
      next_xs,_=unpack(r['iterates'][index+1],tr.nx,tr.nw)
      lam=r['multipliers'][index][tr.nx:].reshape(tr.N,tr.nx+tr.nw-task.nu)
      ld,lf=lam[:,:tr.nx],lam[:,tr.nx:]
      A=np.asarray(linear(jp.asarray(xs[:-1]),jp.asarray(ws)))[:,:,:2]
      C=np.asarray(cf_linear(jp.asarray(xs[:-1]),jp.asarray(ws)))[:,:,:2]
      n,gap=map(np.asarray,geom(jp.asarray(xs[:-1])))
      t=np.stack([-n[:,1],n[:,0]],axis=1)
      gt=-np.einsum('noi,no->ni',A,ld)
      gc=np.einsum('nci,nc->ni',C,lf)
      tangent=np.einsum('ni,ni->n',gt,t)
      step_t=np.einsum('ni,ni->n',next_xs[:-1,:2]-xs[:-1,:2],t)
      # Contributions involving the object translation and object rotation.
      ids=[2,3,task.model.nq+2,task.model.nq+3]
      obj=-np.einsum('noi,no->ni',A[:,ids],ld[:,ids])
      rot=np.zeros_like(obj)
      if label=='box':rot=-np.einsum('noi,no->ni',A[:,[4,9]],ld[:,[4,9]])
      # Largest lateral geometry contribution before the next primal update.
      k=int(np.argmax(abs(tangent)))
      details.append({'iteration':s['history'][index]['iter'],'stage':k,'x':xs[k],'force':ws[k,2],
        'gap':gap[k],'normal':n[k],'tangent':t[k],
        'geometry_lagrangian_tangent':tangent[k],
        'force_equality_lagrangian_tangent':float(gc[k]@t[k]),
        'object_translation_lagrangian_tangent':float(obj[k]@t[k]),
        'object_rotation_lagrangian_tangent':float(rot[k]@t[k]),
        'next_accepted_pusher_step_tangent':step_t[k],
        'summed_geometry_gradient_times_tangent_step':float(tangent@step_t),
        'one_newton_force_tangent_sensitivity':A[k]@t[k]/ws[k,2]})
    all_details[label]=details
  (ROOT/'steering_derivatives.json').write_text(json.dumps(all_details,default=encode,indent=2))
  print(json.dumps(summaries,default=encode,indent=2))

if __name__=='__main__':main()
