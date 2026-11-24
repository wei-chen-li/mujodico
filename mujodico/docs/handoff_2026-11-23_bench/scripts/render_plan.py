"""Render a saved solver plan (results/plans/*.npz, key 'us') under the TRUE law with render_witness.render.
usage: JAX_PLATFORMS=cpu MUJOCO_GL=egl render_plan.py <out_dir> <Task> <seed> <plan.npz> <key>"""
import os, sys, json
OUT, NAME, SEED, PLAN, KEY = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
H = '/home/gchou/PycharmProjects/mujodico-main-dev/mujodico/docs/handoff_2026-11-23_bench/scripts'
sys.path.insert(0, H)
sys.argv = [sys.argv[0], OUT, '__none__']          # render_witness's module-level loop renders nothing
import numpy as np
import render_witness as R
import bench_tasks as T
from simtools import sim_true, metric
t = T.TASKS[NAME](SEED)
us = np.load(PLAN)['us']
xs = sim_true(t, us)
met = metric(t, xs[-1])
n = R.render(t, xs, f'{OUT}/{KEY}.gif', NAME)
print(json.dumps(dict(key=KEY, metric=met, start=metric(t, t.x_0), frames=n, bytes=os.path.getsize(f'{OUT}/{KEY}.gif'))))
