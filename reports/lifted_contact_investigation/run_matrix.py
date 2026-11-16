"""Run the IPOPT ablations in independent processes (three at a time)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = [
    ('sphere_baseline', []),
    ('box_baseline', ['--task', 'PushBox2d']),
    ('sphere_no_linesearch', ['--accept-every']),
    ('sphere_constant', ['--constant-init']),
    ('sphere_free_forces', ['--free-forces']),
    ('sphere_ippas', ['--solver', 'ippas']),
    ('box_fatrop', ['--task', 'PushBox2d', '--solver', 'fatrop']),
    ('sphere_seed1', ['--seed', '1']),
    ('sphere_seed2', ['--seed', '2']),
    ('sphere_local', ['--relu', 'local']),
    ('sphere_softplus', ['--relu', 'softplus']),
    ('sphere_algebraic', ['--relu', 'algebraic']),
    ('sphere_stable_algebraic', ['--relu', 'stable-algebraic']),
    ('sphere_zero', ['--smoothing', '0']),
    ('sphere_small', ['--smoothing', '2e-8']),
    ('sphere_tiny', ['--smoothing', '2e-11']),
    ('sphere_tiny_no_bound_relax', ['--smoothing', '2e-11', '--bound-relax', '0']),
    ('sphere_safe_zero', ['--smoothing', '0', '--safe-zero']),
    ('sphere_small_barrier', ['--mu-init', '1e-8', '--bound-push', '1e-8']),
    ('sphere_accurate_inner', ['--inner-tolerance', '1e-12', '--inner-iterations', '100']),
    ('sphere_exact_hessian', ['--exact-hessian']),
    ('sphere_cone_flag', ['--elliptic']),
    ('sphere_pyramid3', ['--condim', '3']),
    ('sphere_elliptic3', ['--condim', '3', '--elliptic']),
    ('sphere_elliptic3_no_cone', ['--condim', '3', '--elliptic', '--omit-cone']),
    ('sphere_shooting_small', ['--shooting']),
    ('sphere_shooting_default', ['--shooting', '--smoothing', '0.02']),
    ('sphere_consistent_shooting', ['--shooting', '--consistent-smoothing']),
    ('sphere_reduced_barrier', ['--shooting', '--reduced-force-barrier']),
    ('box_no_linesearch', ['--task', 'PushBox2d', '--accept-every']),
    ('box_constant', ['--task', 'PushBox2d', '--constant-init']),
    ('box_free_forces', ['--task', 'PushBox2d', '--free-forces']),
    ('box_algebraic', ['--task', 'PushBox2d', '--relu', 'algebraic']),
    ('box_zero', ['--task', 'PushBox2d', '--smoothing', '0']),
]


def run(case):
  label, args = case
  if (ROOT / 'data' / (label + '.json')).exists():
    return label, 'already saved'
  env = dict(os.environ, JAX_ENABLE_X64='True', JAX_PLATFORMS='cpu',
             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
  with (ROOT / 'data' / (label + '.log')).open('w') as logfile:
    result = subprocess.run([sys.executable, str(ROOT / 'run_experiments.py'),
                             '--label', label, *args], env=env,
                            stdout=logfile, stderr=subprocess.STDOUT)
  return label, result.returncode


if __name__ == '__main__':
  with ThreadPoolExecutor(max_workers=3) as executor:
    for outcome in executor.map(run, CASES):
      print(outcome, flush=True)
