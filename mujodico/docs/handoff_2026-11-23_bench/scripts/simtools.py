"""Rollout / metric helpers for building and certifying benchmark tasks.

sim_c(task, us)      MuJoCo C rollout (soft-constraint contact; fast, for prototyping only)
sim_true(task, us)   the solver's TRUE law (MJX Hunt-Crossley SAP, smoothing 0) on the CPU device --
                     the law every solver result is judged by (same call as lfddp_v.py's read-outs)
metric(task, x)      success metric (task.metric_Q if present, else Q), quaternion-aware; success < 0.01
keyframes(task, kf)  piecewise-linear control targets from [(t_seconds, {actuator_name: target}), ...]
"""
import os
import numpy as np
import mujoco

_TRUE = {}


def metric(task, x):
  from mujodico.examples.methods import util
  Q = np.asarray(getattr(task, 'metric_Q', task.Q))
  return float(util.state_cost(Q, np.asarray(task.x_goal), np.asarray(x), task.quaternion_w_idxs))


def sim_c(task, us, x0=None):
  m = task.model; d = mujoco.MjData(m)
  x0 = task.x_0 if x0 is None else x0
  d.qpos[:] = x0[:m.nq]; d.qvel[:] = x0[m.nq:m.nq + m.nv]
  xs = [np.concatenate([d.qpos, d.qvel])]
  for u in us:
    d.ctrl[:] = u
    mujoco.mj_step(m, d)
    xs.append(np.concatenate([d.qpos, d.qvel]))
  return np.array(xs)


def sim_true(task, us, x0=None):
  """True-law rollout on the CPU device.  Returns states (N+1, nx)."""
  import jax
  from mujodico.examples.methods import IppasLiftedMultipleShooting
  key = (type(task).__name__, task.horizon)
  cpu = jax.devices('cpu')[0]
  with jax.default_device(cpu):
    if key not in _TRUE:
      _TRUE[key] = IppasLiftedMultipleShooting(task)
    tm = _TRUE[key]
    tm.update(task)
    x0 = task.x_0 if x0 is None else x0
    X, _ = tm.rollout(np.asarray(x0, np.float64), np.asarray(us, np.float64), 0.0)   # smoothing x0 = true law
    return np.asarray(X)


def keyframes(task, kf, dt=None):
  """kf: list of (t, {actuator_name: ABSOLUTE target}); unspecified actuators hold their previous target
  (initial target = the start pose).  Linear interpolation between keyframes, clipped to the actuator
  ranges; returns (N, nu) controls in the task's convention (relative to task.u_offset if it has one)."""
  m = task.model
  dt = m.opt.timestep if dt is None else dt
  off = np.asarray(getattr(task, 'u_offset', np.zeros(m.nu)), float)
  names = [m.actuator(i).name for i in range(m.nu)]
  ts = [0.0]; vals = [np.asarray(task.u_ref, float) + off]
  for t, tgt in kf:
    v = vals[-1].copy()
    tgt = dict(tgt)
    if 'a_grip' in tgt and 'a_grip_r' in names and 'a_grip_r' not in tgt:   # independent jaws: mirror the grip
      tgt['a_grip_r'] = tgt['a_grip']
    for k, val in tgt.items():
      v[names.index(k)] = val
    ts.append(t); vals.append(v)
  ts = np.array(ts); vals = np.array(vals)
  tk = np.arange(task.horizon) * dt
  us = np.stack([np.interp(tk, ts, vals[:, i]) for i in range(m.nu)], 1)
  lb, ub = task.u_bounds
  return np.clip(us - off, lb, ub)


def report(task, us, which=('c', 'true'), verbose=True):
  out = {}
  for w in which:
    X = sim_c(task, us) if w == 'c' else sim_true(task, us)
    out[w] = (metric(task, X[-1]), X)
    if verbose:
      print(f'  [{w:4s}] metric {out[w][0]:.4g}  finite {np.isfinite(X).all()}')
  return out
