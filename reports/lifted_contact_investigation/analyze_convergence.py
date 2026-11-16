"""Summarize KKT diagnostics and draw residual/continuation figures."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'convergence'
DATA=OUT/'data'
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)
labels=['sphere_baseline','box_baseline','circular_baseline','square_baseline']

rows=[]
for label in labels:
  s=json.loads((DATA/(label+'_kkt.json')).read_text())
  rows.append({k:s[k] for k in ['label','task','dynamics_defect_max','force_equality_defect_max',
    'stationarity_max_unscaled','complementarity_max_relaxed_bounds','perturbed_complementarity_max',
    'barrier_mu','bounds_violation_max','force_rows_exceeding_1e_minus4']})
with (OUT/'baseline_residuals.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

rows=[]
for p in sorted(DATA.glob('*.json')):
  if p.name.endswith('_history.json'):continue
  s=json.loads(p.read_text())
  if 'options' not in s:continue
  last=s['last_iteration'];v=last['unscaled_violations']
  rows.append({'label':p.stem,'source':s['args']['source'],'mode':s['args']['mode'],
    'checkpoint':s['args'].get('checkpoint'),'tight_inner':s['args']['inner_tolerance'] is not None,
    'status':s['info']['status'],'iterations':last['iter'],
    'dynamics_defect':s['dynamics_defect_max'],'force_defect':s['force_equality_defect_max'],
    'stationarity':v['grad_lag_x'],'bound_complementarity':max(v['compl_x_L'],v['compl_x_U']),
    'mu':last['mu'],'objective':s['info']['obj_val']})
with (OUT/'polishing_results.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,label):
  fig.savefig(FIG/(label+'.png'),dpi=180,bbox_inches='tight')
  fig.savefig(FIG/(label+'.pdf'),bbox_inches='tight')
  plt.close(fig)

fig,axes=plt.subplots(4,2,figsize=(11,11),constrained_layout=True)
for row,label in enumerate(labels):
  raw=np.load(DATA/(label+'_residuals.npz'))
  s=json.loads((ROOT/'data'/(label+'.json')).read_text())
  ncf=s['ncf']; n=len(raw['force']);nx=raw['dynamics'].shape[1];nw=2+ncf
  dt=s.get('timestep',.05 if label=='box_baseline' else .02)
  time=np.arange(n)*dt
  ax=axes[row,0]
  ax.semilogy(time,np.maximum(np.max(np.abs(raw['force']),axis=1),1e-16),color='#cf5439',label='Force equality (N)')
  ax.semilogy(time,np.maximum(np.max(np.abs(raw['dynamics']),axis=1),1e-16),color='#226eaa',label='State continuity (mixed units)')
  ax.axhline(1e-4,color='k',ls=':',lw=1,label='Absolute primal guard')
  ax.set(title=label.replace('_baseline','').capitalize(),xlabel='Stage time (s)',ylabel='Maximum stage defect')
  z=raw['stationarity'];stage=np.r_[z[:n*(nx+nw)].reshape(n,nx+nw),np.pad(z[n*(nx+nw):][None],((0,0),(0,nw)))]
  axes[row,1].semilogy(np.arange(n+1)*dt,np.maximum(np.max(abs(stage),axis=1),1e-16),color='#8c539e')
  axes[row,1].set(xlabel='State/stage time (s)',ylabel='Maximum stationarity residual')
  for a in axes[row]:a.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
save(fig,'residual_localization')

fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
cases=[ [('sphere_baseline_polish_gn','Objective Hessian only'),
          ('sphere_baseline_polish_exact','Full Lagrangian Hessian')],
        [('circular_polish_more','Full Hessian, default inner accuracy'),
         ('circular_polish_tight','Full Hessian, tight inner accuracy')]]
for row,group in enumerate(cases):
  for label,desc in group:
    h=json.loads((DATA/(label+'_history.json')).read_text())
    # Restoration dual residuals describe the restoration problem; omit them
    # when comparing stationarity of the original NLP.
    h=[q for q in h if q['alg_mod']==0]
    its=np.asarray([q['iter'] for q in h])
    values=[[q['unscaled_violations']['g_violation'] for q in h],
            [q['unscaled_violations']['grad_lag_x'] for q in h],
            [max(q['unscaled_violations']['compl_x_L'],q['unscaled_violations']['compl_x_U']) for q in h]]
    for ax,y in zip(axes[row],values):ax.plot(its,np.maximum(y,1e-16),label=desc,lw=1)
  for ax,title in zip(axes[row],['Primal constraint violation','Lagrangian stationarity','Bound complementarity']):
    ax.set(xscale='symlog',yscale='log',xlabel='Additional iterations',title=title)
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
axes[0,0].set_ylabel('Sphere')
axes[1,0].set_ylabel('Circular knob: same checkpoint')
save(fig,'polishing_comparison')
trace_path=DATA/'square_feature_trace_history.json'
if trace_path.exists():
  h=json.loads(trace_path.read_text());it=[q['iter'] for q in h]
  fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
  axes[0].plot(it,[q['trace']['projection_margins'][0]*1e6 for q in h],'-o',ms=3,color='#226eaa')
  axes[0].axhline(0,color='k',ls=':',lw=1)
  axes[0].set(ylabel='Projection margin (micrometres)',title='Square knob: repeated face/corner transitions at stage 107')
  axes[1].plot(it,[q['trace']['pusher_y_stationarity'] for q in h],'-o',ms=3,color='#cf5439')
  axes[1].axhline(0,color='k',ls=':',lw=1)
  axes[1].set(xlabel='Additional iteration',ylabel='Pusher y stationarity residual')
  for ax in axes:ax.grid(alpha=.2)
  save(fig,'square_feature_cycle')
print('Wrote convergence summary CSV files and report figure pairs.')
