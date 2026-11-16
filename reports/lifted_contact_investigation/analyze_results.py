"""Build the comparison table and publication-ready static report figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

root = Path(__file__).resolve().parent
data = root / 'data'
figures = root / 'figures'
figures.mkdir(exist_ok=True)
summaries = {}
for path in sorted(data.glob('*.json')):
  if path.name.endswith('_history.json') or 'probes' in path.name:
    continue
  summary = json.loads(path.read_text())
  if not isinstance(summary, dict) or 'args' not in summary:
    continue
  summaries[path.stem] = summary

rollouts = json.loads((data / 'rollout_probes.json').read_text())
rows = []
for name, s in summaries.items():
  a, info = s['args'], s['info']
  n = s.get('horizon', 100 if a['task'] == 'PushSphere2d' else 80)
  nx = s.get('nx', 8 if a['task'] == 'PushSphere2d' else 10)
  dynamics_defect = s.get('dynamics_defect_max')
  force_defect = s.get('force_equality_defect_max')
  if 'g' in info and dynamics_defect is None:
    g = np.asarray(info['g'])[nx:].reshape(n, -1)
    dynamics_defect = float(np.max(np.abs(g[:, :nx])))
    if s['ncf']:
      force_defect = float(np.max(np.abs(g[:, nx:nx+s['ncf']])))
  status = info['status']
  solver = a.get('solver', 'ipopt')
  success = (status in [0, 1]) if solver == 'ipopt' else bool(info.get('success', False))
  row = dict(label=name, task=a['task'], solver=solver, seed=a['seed'],
             smoothing=a['smoothing'], status=status, success=success,
             iterations=s['iterations'] or info.get('iterations'),
             objective=s['final_objective'],
             planned_position_error_m=s['object_position_error'],
             planned_hinge_error_radians=s.get('hinge_angle_error_radians'),
             smooth_replay_hinge_error_radians=rollouts.get(name, {}).get('smooth_replay_hinge_error_radians'),
             zero_smoothing_replay_hinge_error_radians=s.get('replay_hinge_angle_error_radians'),
             smooth_replay_position_error_m=rollouts.get(name, {}).get('smooth_replay_position_error'),
             zero_smoothing_replay_position_error_m=s['replay_object_position_error'],
             constraint_violation_max=s['constraint_violation_max'],
             dynamics_defect_max=dynamics_defect, force_equality_defect_max=force_defect,
             relative_angle_range_degrees=s['relative_angle_range_degrees'],
             backtracked_iterations=s['backtracked_iterations'] if solver == 'ipopt' else None,
             fractional_primal_steps=s['fractional_primal_steps'] if solver == 'ipopt' else None)
  rows.append(row)
with (root / 'results.csv').open('w') as f:
  writer = csv.DictWriter(f, fieldnames=list(rows[0]))
  writer.writeheader()
  writer.writerows(rows)
environment_path = data / 'environment.json'
if environment_path.exists():
  environment = json.loads(environment_path.read_text())
  environment.pop('successful_output_runs', None)
  environment['completed_experiment_runs'] = len(rows)
  environment_path.write_text(json.dumps(environment, indent=2))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 130})


def save(fig, name):
  fig.savefig(figures / (name + '.png'), dpi=180, bbox_inches='tight')
  fig.savefig(figures / (name + '.pdf'), bbox_inches='tight')
  plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for ax, name, title in zip(axes.flat,
    ['sphere_baseline', 'box_baseline', 'sphere_constant', 'sphere_free_forces'],
    ['Sphere: default lifted initialization', 'Box: default lifted initialization',
     'Sphere: constant-state initialization', 'Sphere: force bounds removed']):
  a = np.load(data / (name + '.npz'))
  x = a['xs']
  ax.plot(x[:, 0], x[:, 1], color='#d04c38', label='Pusher, optimized')
  ax.plot(x[:, 2], x[:, 3], color='#226eaa', lw=2, label='Object, optimized')
  ax.scatter(*x[0, :2], color='#d04c38', marker='s', zorder=4)
  ax.scatter(*x[0, 2:4], color='#226eaa', marker='s', zorder=4)
  ax.scatter(0, 0, marker='*', s=140, color='#16a079', label='Object goal')
  if 'box' not in name:
    ax.add_patch(Circle(x[0, 2:4], .08, color='#226eaa', alpha=.12))
    ax.add_patch(Circle(x[-1, :2], .05, color='#d04c38', alpha=.25))
  else:
    ax.add_patch(Rectangle(x[0,2:4]-[.15,.08], .30,.16,
                           angle=x[0,4]*180/np.pi, rotation_point='center',
                           color='#226eaa',alpha=.15))
  ax.set(title=title, xlabel='x (m)', ylabel='y (m)', xlim=(-.25,.55), ylim=(-.3,.55))
  ax.set_aspect('equal')
  ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8, loc='upper left')
save(fig, 'paths_and_initialization')

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
for name, label, color in [('sphere_baseline','Default IPOPT','#226eaa'),
                           ('sphere_no_linesearch','Accept every trial','#d04c38')]:
  h = json.loads((data / (name + '_history.json')).read_text())
  it = [a['iter'] for a in h]
  for ax, key in zip(axes.flat,['objective','inf_pr','mu','alpha_pr']):
    ax.plot(it, [a[key] for a in h], label=label, color=color, lw=1, alpha=.8)
axes[0,0].set(ylabel='Objective', xlabel='Iteration')
axes[0,1].set(ylabel='Equality defect (maximum)', xlabel='Iteration', yscale='log')
axes[1,0].set(ylabel='Barrier parameter μ', xlabel='Iteration', yscale='log')
axes[1,1].set(ylabel='Primal step fraction', xlabel='Iteration')
axes[0,0].legend()
for ax in axes.flat:
  ax.grid(alpha=.2)
save(fig, 'line_search_and_feasibility')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
p = json.loads((data / 'derivative_probes.json').read_text())
for task, label, color in [('PushSphere2d','Sphere','#226eaa'),('PushBox2d','Box','#d04c38')]:
  r = [a for a in p[task] if a['smoothing'] > 0]
  k = [a['smoothing'] for a in r]
  axes[0].loglog(k,[a['force'][0] for a in r],'-o',label=label,color=color)
  axes[1].semilogx(k,[np.linalg.norm(a['log_force_gradient_pusher']) for a in r],'-o',label=label,color=color)
  axes[1].semilogx(k,[np.linalg.norm(a['relaxed_log_force_gradient_pusher']) for a in r],'--',color=color,
                   label=label+' with 1e-8 bound relaxation')
axes[0].set(xlabel='Contact smoothing κ',ylabel='Initial contact force (N)', title='Distant force vanishes with smoothing')
axes[1].set(xlabel='Contact smoothing κ',ylabel='‖∂ log(force) / ∂ pusher position‖ (1/m)',
            title='Normalized gradient survives in exact arithmetic')
for ax in axes:
  ax.grid(alpha=.2)
  ax.legend(fontsize=8)
save(fig, 'smoothing_and_log_gradient')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, label in zip(axes,['sphere_baseline','box_baseline']):
  a = np.load(data/(label+'.npz'))
  smooth = np.load(data/(label+'_smooth_replay.npz'))['xs']
  for x, desc, style in [(a['xs'],'Optimized states','-'),
                          (smooth,'Same controls, smoothed replay','--'),
                          (a['replay_xs'],'Same controls, zero-smoothing replay',':')]:
    ax.plot(x[:,2],x[:,3],style,label=desc,lw=2)
    ax.scatter(*x[-1,2:4],s=25)
  ax.scatter(0,0,marker='*',s=120,color='#16a079',label='Goal')
  ax.set(xlabel='Object x (m)',ylabel='Object y (m)',title=label.replace('_',' ').capitalize())
  ax.set_aspect('equal', adjustable='datalim')
  ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
save(fig, 'optimized_and_replayed_objects')

if all((data / (name+'_smooth_replay.npz')).exists()
       for name in ['circular_baseline', 'square_baseline']):
  fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
  for row, label, title in [(0, 'circular_baseline', 'Circular knob'),
                            (1, 'square_baseline', 'Square knob')]:
    a = np.load(data/(label+'.npz'))
    smooth = np.load(data/(label+'_smooth_replay.npz'))['xs']
    t = np.arange(len(a['xs']))*.02
    for x, desc, style in [(a['xs'], 'Optimized states', '-'),
                            (smooth, 'Smoothed replay', '--'),
                            (a['replay_xs'], 'Zero-smoothing replay', ':')]:
      axes[row,0].plot(t, np.degrees(x[:,2]), style, label=desc, lw=1.8)
    axes[row,0].axhline(0, color='#16a079', alpha=.6)
    axes[row,0].set(xlabel='Time (s)', ylabel='Unwrapped hinge angle (degrees)', title=title)
    axes[row,0].legend(fontsize=8)
    axes[row,1].plot(a['xs'][:,0],a['xs'][:,1],color='#d04c38',label='Optimized pusher')
    axes[row,1].scatter(*a['xs'][0,:2],color='#d04c38',marker='s',label='Start')
    if row == 0:
      axes[row,1].add_patch(Circle((0,0),.08,color='#226eaa',alpha=.2))
    else:
      axes[row,1].add_patch(Rectangle((-.08,-.08), .16,.16,
                           angle=np.degrees(a['xs'][0,2]), rotation_point='center',
                           color='#226eaa',alpha=.2))
    axes[row,1].set(xlabel='Pusher x (m)',ylabel='Pusher y (m)',title='Contact acquisition and turning')
    axes[row,1].set_aspect('equal')
    for ax in axes[row]:
      ax.grid(alpha=.2)
  save(fig, 'frictional_knob_turning')
print(f'Wrote {len(rows)} experiment rows and report figures.')
