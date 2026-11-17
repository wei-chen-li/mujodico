"""Build comparisons for the same-height, upward-pointing finger layout."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from run import ROOT,DATA,dump

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)

def segments(x,side):
  base=np.array([-.15 if side==0 else .15,-.25]);a,b=x[2*side:2*side+2]
  d1=np.array([-np.sin(a),np.cos(a)]);d2=np.array([-np.sin(a+b),np.cos(a+b)])
  joint=base+.15*d1
  return base,joint,joint+.06*d2,joint+.15*d2

def gap(x,side):
  a,b,c,d=segments(x,side);gs=[]
  for p,q in [(a,b),(c,d)]:
    v=q-p;t=np.clip(-p@v/(v@v+1e-6),0,1)
    gs.append(np.linalg.norm(p+t*v)-.11)
  return min(gs)

def events(g):
  result=[];contact=False;released=None
  for k,v in enumerate(g):
    if v<=.0005:
      if released is not None and k-released>=3:
        result.append({'release':released*.02,'recontact':k*.02,
          'maximum_gap_mm':float(1000*max(g[released:k]))})
      contact=True;released=None
    elif contact and v>=.002 and released is None:released=k
  return result

def main():
  results={};rows=[]
  for p in DATA.glob('*.json'):
    s=json.loads(p.read_text());a=s['args'];r=np.load(p.with_suffix('.npz'))
    nx=s['nx'];nq=nx//2;nu=s['nu'];x,w=r['xs'],r['ws']
    qslack=float((np.pi/2-abs(x[:,:nu])).min()) if a['task'] in ['finger','two'] else None
    uslack=float(((2 if a['task'] in ['finger','two'] else .2)-abs(w[:,:nu])).min())
    info=s['info'];last=s['history'][-1] if s['history'] else {}
    row={'label':p.stem,'task':a['task'],'same_joint_angles':a.get('same_joint_angles',False),
      'mirror_axis':a.get('mirror_axis','y'),'finger_collisions':a.get('finger_collisions',False),
      'goal_degrees':a['goal_degrees'],'solver':a['solver'],'force_floor':a['force_floor'],
      'mu_init':a.get('mu_init',.1),'objective':s['objective'],'angle':s['angle_degrees'],
      'smooth_replay_angle':s['replay']['smooth']['angle_degrees'],
      'zero_replay_angle':s['replay']['zero']['angle_degrees'],'violation':s['violation'],
      'status':info['status'],'iterations':last.get('iter',info.get('iter_count')),
      'dual_error':last.get('inf_du',info.get('dual_error')),'final_mu':last.get('mu',info.get('mu')),
      'joint_limit_slack':qslack,'control_limit_slack':uslack,
      'force_bound_slack':float(w[:,nu:].min()+a['force_floor']) if w.shape[1]>nu else None,
      'smooth_replay_bound_violation':s['replay']['smooth']['bound_violation'],
      'zero_replay_bound_violation':s['replay']['zero']['bound_violation']}
    rows.append(row);results[p.stem]=s
  with (ROOT/'results.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

  choices=[('quadrant','Cartesian, third quadrant'),('finger','One finger'),
    ('two','Two fingers, opposite bends'),('two_copied','Copied bends, IPOPT'),
    ('two_copied_ippas','Copied bends, IPPAS'),('two_copied_mu_ippas','Copied bends, IPPAS, mu=0.01')]
  fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
  for label,title in choices:
    if label not in results:continue
    x=np.load(DATA/(label+'.npz'))['smooth_xs'];nq=x.shape[1]//2
    ax.plot(np.arange(len(x))*.02,np.degrees(x[:,nq-1]),label=title)
  ax.axhline(360,color='k',ls=':',label='Goal');ax.legend(fontsize=8)
  ax.set(xlabel='Physical replay time (s)',ylabel='Knob angle (degrees)');ax.grid(alpha=.2)
  for ext in ['png','pdf']:fig.savefig(FIG/('rotation_comparison.'+ext),dpi=180)
  plt.close(fig)

  fig,axs=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
  for ax,label,title in zip(axs,['finger','two','two_copied'],['Native one finger','Two: opposite bends','Two: copied bends']):
    x=np.load(DATA/(label+'.npz'))['xs'][0]
    for side,color in zip(range(1 if label=='finger' else 2),['tab:blue','tab:orange']):
      a,b,c,d=segments(x,side)
      for p,q in [(a,b),(c,d)]:
        v=q-p;n=np.array([-v[1],v[0]])/np.linalg.norm(v)*.03
        ax.add_patch(Polygon([p+n,q+n,q-n,p-n],color=color,alpha=.8))
        ax.add_patch(Circle(p,.03,color=color,alpha=.8));ax.add_patch(Circle(q,.03,color=color,alpha=.8))
      ax.plot([b[0],c[0]],[b[1],c[1]],color=color)
      ax.scatter(a[0],a[1],s=20,c='k')
    ax.add_patch(Circle((0,0),.08,color='gray',alpha=.8))
    ax.set(aspect='equal',xlim=(-.32,.32),ylim=(-.31,.14),title=title,xlabel='x (m)',ylabel='y (m)')
    ax.grid(alpha=.2)
  fig.savefig(FIG/'initial_configurations.png',dpi=180);plt.close(fig)

  event_data={}
  fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
  for label,color in [('quadrant','tab:green'),('finger','tab:blue'),('two','tab:orange')]:
    x=np.load(DATA/(label+'.npz'))['smooth_xs'];t=np.arange(len(x))*.02
    gs=np.linalg.norm(x[:,:2],axis=1)-.1 if label=='quadrant' else np.array([gap(v,0) for v in x])
    event_data[label]=events(gs)
    axs[0].plot(t,gs*1000,label=label,color=color)
  axs[0].set(ylabel='Whole-pusher gap (mm)',ylim=(-1,15));axs[0].legend()
  for j,label in enumerate(['finger','two']):
    x=np.load(DATA/(label+'.npz'))['smooth_xs'];t=np.arange(len(x))*.02
    axs[1+j].plot(t,np.degrees(x[:,0]),label=label+' q1')
    axs[1+j].plot(t,np.degrees(x[:,1]),label=label+' q2')
    axs[1+j].axhline(-90,color='k',ls=':');axs[1+j].axhline(90,color='k',ls=':')
    axs[1+j].set(ylabel='Joint angle (degrees)');axs[1+j].legend()
  for ax in axs:ax.grid(alpha=.2)
  axs[-1].set_xlabel('Physical replay time (s)')
  fig.savefig(FIG/'contact_and_joint_cycles.png',dpi=180);plt.close(fig)
  dump(ROOT/'proximity_events.json',event_data)
  print('Saved',len(rows),'completed experiments.')

if __name__=='__main__':main()
