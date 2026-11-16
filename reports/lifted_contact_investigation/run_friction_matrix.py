"""Native-friction knob experiments, including matched cone ablations."""
from concurrent.futures import ThreadPoolExecutor
from run_matrix import run

CASES = [
    ('circular_baseline', ['--task', 'PushCircularKnob', '--smoothing', '1e-5']),
    ('square_baseline', ['--task', 'PushSquareKnob', '--smoothing', '1e-5']),
    ('circular_elliptic', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--elliptic']),
    ('square_elliptic', ['--task', 'PushSquareKnob', '--smoothing', '1e-5', '--elliptic']),
    ('circular_free_forces', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--free-forces']),
    ('square_free_forces', ['--task', 'PushSquareKnob', '--smoothing', '1e-5', '--free-forces']),
    ('circular_algebraic', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--relu', 'algebraic']),
    ('square_algebraic', ['--task', 'PushSquareKnob', '--smoothing', '1e-5', '--relu', 'algebraic']),
    ('circular_elliptic_no_cone', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--elliptic', '--omit-cone']),
    ('square_elliptic_no_cone', ['--task', 'PushSquareKnob', '--smoothing', '1e-5', '--elliptic', '--omit-cone']),
    ('circular_no_linesearch', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--accept-every']),
    ('circular_ippas', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--solver', 'ippas']),
    ('circular_frictionless', ['--task', 'PushCircularKnob', '--smoothing', '1e-5', '--condim', '1']),
]

if __name__ == '__main__':
  with ThreadPoolExecutor(max_workers=3) as executor:
    for result in executor.map(run, CASES):
      print(result, flush=True)
