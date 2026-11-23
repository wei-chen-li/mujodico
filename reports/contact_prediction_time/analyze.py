"""Build tables and figures from the saved prediction-time experiments."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mujodico-contact-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIG = ROOT / 'figures'


def angle_span(xs):
    delta = xs[:, :2] - xs[:, 2:4]
    return float(np.rad2deg(np.ptp(np.unwrap(np.arctan2(delta[:, 1], delta[:, 0])))))


def unpack(z, d):
    n, nx, nw = d['N'], d['nx'], d['nw']
    stage = z[:-nx].reshape(n, nx + nw)
    return np.vstack([stage[:, :nx], z[-nx:]])


def main():
    FIG.mkdir(exist_ok=True)
    results = {p.stem: json.loads(p.read_text()) for p in sorted(DATA.glob('*.json'))}
    rows = []
    for label, d in results.items():
        a, h, info = d['args'], d['history'], d['info']
        raw = np.load(DATA / (label + '.npz'))
        last = h[-1] if h else info
        row = dict(label=label, task=a['task'], solver=a['solver'], form=a.get('form', 'lifted'),
                   alpha=a['alpha'], width_alpha=a['width_alpha'] or a['alpha'],
                   manual_alpha=a['manual_alpha'] or a['alpha'],
                   smoothing=a['smoothing'], initialization=a['init'] or 'linear',
                   hessian=a['hessian'], inner_tolerance=a['inner_tolerance'],
                   mu_target=a['mu_target'], status=info['status'],
                   iterations=last.get('iter', info.get('iter_count')),
                   objective=d['objective'], planned_error_m=d['goal_error'],
                   smooth_replay_error_m=d['replay']['smooth']['position_error'],
                   zero_replay_error_m=d['replay']['zero']['position_error'],
                   planned_angle_span_deg=d['angle_span_degrees'],
                   smooth_angle_span_deg=angle_span(raw['smooth_xs']),
                   force_defect_N=d['force_defect'], dynamics_defect=d['dynamics_defect'],
                   constraint_violation=d['constraint_violation'],
                   final_mu=last.get('mu'), final_primal=last.get('inf_pr', info.get('primal_error')),
                   final_dual=last.get('inf_du', info.get('dual_error')),
                   restoration_callbacks=sum(e['alg_mod'] == 1 for e in h))
        rows.append(row)
    with (ROOT / 'results.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    probes = json.loads((ROOT / 'probes.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for label, color, title in [('a2', '#1565c0', r'$\tau=h$'), ('a1', '#d55e00', r'$\tau=0$')]:
        c = probes[label]['force_curve']
        axes[0].semilogy(np.array(c['gaps'])*1000, c['forces'], color=color, label=title)
        axes[1].plot(np.array(c['gaps'])*1000, np.array(c['next_gaps'])*1000, color=color, label=title)
    for ax in axes:
        ax.set_xlabel('Initial clearance (mm)')
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Self-consistent normal force (N)')
    axes[1].set_ylabel('Clearance after the actual step (mm)')
    fig.suptitle('Identical current states and 0.8 m/s closing command; native sphere model')
    fig.savefig(FIG / 'anticipatory_force.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.7), constrained_layout=True)
    for ax, (label, title) in zip(axes.flat, [
            ('sphere_reduced_a2', 'Sphere, reduced, tau=h'),
            ('sphere_reduced_a1', 'Sphere, reduced, tau=0'),
            ('sphere_a1', 'Sphere, lifted, tau=0'),
            ('box_reduced_a2', 'Box, reduced, tau=h'),
            ('box_reduced_a1', 'Box, reduced, tau=0'),
            ('box_a1_holdmu', 'Box, lifted, tau=0, mu target=0.02')]):
        raw = np.load(DATA / (label + '.npz'))
        x = raw['xs']
        replay = raw['smooth_xs']
        ax.plot(x[:, 0], x[:, 1], color='#c44e52', label='Pusher: planned')
        ax.plot(x[:, 2], x[:, 3], color='#1565c0', label='Object: planned')
        ax.plot(replay[:, 2], replay[:, 3], '--', color='#009e73', label='Object: replay')
        ax.scatter(x[0, [0, 2]], x[0, [1, 3]], marker='s', c=['#c44e52', '#1565c0'], s=18)
        ax.scatter([0], [0], c='black', marker='x', s=35, label='Object goal')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.savefig(FIG / 'routes.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    progress = {}
    for col, (task, forms) in enumerate([
            ('sphere', [('reduced_a2', '#1565c0'), ('reduced_a1', '#d55e00'), ('a1', '#009e73')]),
            ('box', [('reduced_a2', '#1565c0'), ('reduced_a1', '#d55e00'), ('a1_holdmu', '#009e73')])]):
        for suffix, color in forms:
            label = task + '_' + suffix
            d = results[label]
            raw = np.load(DATA / (label + '.npz'))
            hs = d['history']
            it = np.array([h['iter'] for h in hs])
            spans = [angle_span(unpack(z, d)) for z in raw['iterates']]
            name = ('Reduced, tau=h' if suffix == 'reduced_a2' else
                    'Reduced, tau=0' if suffix == 'reduced_a1' else
                    'Lifted, tau=0' + (', held mu' if task == 'box' else ''))
            axes[0, col].plot(it, spans, c=color, label=name)
            axes[1, col].semilogy(it, [h['mu'] for h in hs], c=color)
            progress[label] = [{'iteration': int(i), 'angle_span_degrees': s,
                                'mu': h['mu'], 'inf_pr': h['inf_pr'], 'inf_du': h['inf_du']}
                               for i, s, h in zip(it, spans, hs)]
        axes[0, col].set_title(task.capitalize())
        axes[0, col].set_ylabel('Planned relative-angle span (degrees)')
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_ylabel('Barrier parameter mu')
        for ax in axes[:, col]:
            ax.set_xlabel('Interior-point iteration')
            ax.grid(alpha=.2)
    fig.savefig(FIG / 'discovery_and_barrier.png', dpi=180)
    plt.close(fig)
    (ROOT / 'progress.json').write_text(json.dumps(progress, indent=2))
    print(f'Wrote {len(rows)} result rows and three figures.')


if __name__ == '__main__':
    main()
