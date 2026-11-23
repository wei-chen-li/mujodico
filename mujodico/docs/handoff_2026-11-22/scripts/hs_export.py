"""On-disk jax.export cache for the resumable homotopy IP loop (HomotopyIpPas._init / _loop).

The package caches its one-shot IP-PAS solve this way (IpPasSolver._exported_fn); PLF calls the resumable loop
instead, which every process re-traced (~12 s for _loop + ~5 s for _init on PushT2dHard).  The export is keyed on
the package fingerprint (solver.export_key: package source, planning model, task arrays, settings), the source of
homotopy.py / ippas_resume.py / this file, the argument shapes, the JAX version and the backend.  Same StableHLO, so
the same iterates; set PLF_HS_EXPORT=0 to disable."""
import os, hashlib, pickle, dataclasses
import jax, jax.numpy as jnp
from mujodico.trajopt._src.fatrop_ref import ippas
from homotopy import HomotopyIpPas

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = hashlib.sha256(b''.join(open(os.path.join(_HERE, f_), 'rb').read()
                               for f_ in ('homotopy.py', 'ippas_resume.py', 'hs_export.py'))).hexdigest()


class ExportedHomotopyIpPas(HomotopyIpPas):

  def __init__(self, solver, *a, **k):
    super().__init__(solver, *a, **k)
    key = getattr(solver, 'export_key', None)
    self._xdir = solver.export_dir if (key and solver.export_dir and os.environ.get('PLF_HS_EXPORT', '1') != '0') else None
    self._xkey = key
    self._xfn = {}

  def _exported(self, name, impl, args, opts=True):
    leaves, in_def = jax.tree.flatten(args)
    leaves = [jnp.asarray(l) for l in leaves]
    # in-memory key without weak types (as Resumable._run: the loop compiled for the first call's avals -- init's weak
    # scalars -- is reused when later calls pass strong ones); the disk key includes the first call's weak types
    spec = repr(in_def) + repr([(tuple(l.shape), str(l.dtype)) for l in leaves])
    ck = (name, spec)
    if ck in self._xfn:
      return self._xfn[ck](leaves)
    wk = repr([bool(getattr(l, 'weak_type', False)) for l in leaves])
    h = hashlib.sha256('|'.join((self._xkey, _SRC, name, spec, wk, jax.__version__, jax.default_backend())).encode()).hexdigest()[:40]
    path = os.path.join(self._xdir, f'hs_{h}.jaxexp.pkl')
    got = None
    if os.path.exists(path):
      try:
        with open(path, 'rb') as f:
          got = pickle.load(f)
      except Exception:
        got = None
    if got is None:
      box = {}
      def flat(*ls):
        out = impl(*jax.tree.unflatten(in_def, list(ls)))
        o_leaves, box['out_def'] = jax.tree.flatten(out)
        return o_leaves
      exp = jax.export.export(jax.jit(flat))(*[jax.ShapeDtypeStruct(l.shape, l.dtype, weak_type=getattr(l, 'weak_type', False))
                                               for l in leaves])
      got = (dataclasses.replace(exp, _get_vjp=None), box['out_def'])
      os.makedirs(self._xdir, exist_ok=True)
      tmp = f'{path}.{os.getpid()}.tmp'
      with open(tmp, 'wb') as f:
        f.write(pickle.dumps(got))
      os.replace(tmp, path)
    exp, out_def = got
    low = jax.jit(exp.call).lower(*leaves)
    co = ippas.compile_options() if opts else None   # as the parent: _loop with the solver's options, _init plain jit
    comp = low.compile(compiler_options=co) if co else low.compile()
    fn = lambda ls: jax.tree.unflatten(out_def, comp(*ls))
    self._xfn[ck] = fn
    return fn(leaves)

  def start(self, params, x_init, w_init, chunk_end, max_iter=None):
    if self._xdir is None:
      return super().start(params, x_init, w_init, chunk_end, max_iter)
    mi = self.sv.opt.max_iter if max_iter is None else int(max_iter)
    if mi > self.sv.opt.max_iter:
      raise ValueError('max_iter exceeds the compiled IpPasOptions.max_iter')
    S = self._exported('init', self._init_impl, (params, jnp.asarray(x_init, jnp.float64), jnp.asarray(w_init, jnp.float64),
                                                 jnp.asarray(mi)), opts=False)
    return self._run(params, S, chunk_end)

  def _run(self, params, S, chunk_end):
    if self._xdir is None:
      return super()._run(params, S, chunk_end)
    return self._exported('loop', self._loop_impl, (params, S, jnp.asarray(chunk_end)))


class ExportedJit:
  """A jitted function served from an on-disk jax.export artifact: tracing and lowering happen once per argument
  spec and platform across processes (the compile itself still goes through JAX's persistent cache).  Same StableHLO,
  so the same results.  `key` must identify everything the function closes over (model, task structure, settings,
  source).  Arguments must be arrays (or pytrees of arrays) committed to one device."""

  def __init__(self, fn, name, key, xdir):
    self.fn, self.name, self.key, self.xdir = fn, name, key, xdir
    self._c = {}

  def __call__(self, *args):
    leaves, in_def = jax.tree.flatten(args)
    leaves = [jnp.asarray(l) for l in leaves]
    devs = {d for l in leaves for d in l.devices()}
    dev = next(iter(devs)) if len(devs) == 1 else None
    spec = repr(in_def) + repr([(tuple(l.shape), str(l.dtype)) for l in leaves])
    ck = (spec, str(dev))
    hit = self._c.get(ck)
    if hit is not None:
      return hit(leaves)
    if dev is None or self.xdir is None:   # mixed devices: plain jit
      return self.fn(*args)
    plat = {'gpu': 'cuda'}.get(dev.platform, dev.platform)   # jax.export names the CUDA platform 'cuda'
    wk = repr([bool(getattr(l, 'weak_type', False)) for l in leaves])
    h = hashlib.sha256('|'.join((self.key, self.name, spec, wk, plat, jax.__version__)).encode()).hexdigest()[:40]
    path = os.path.join(self.xdir, f'fn_{h}.jaxexp.pkl')
    got = None
    if os.path.exists(path):
      try:
        with open(path, 'rb') as f:
          got = pickle.load(f)
      except Exception:
        got = None
    if got is None:
      box = {}
      fn = self.fn
      def flat(*ls):
        out = fn(*jax.tree.unflatten(in_def, list(ls)))
        o, box['out_def'] = jax.tree.flatten(out)
        return o
      exp = jax.export.export(jax.jit(flat), platforms=(plat,))(
          *[jax.ShapeDtypeStruct(l.shape, l.dtype, weak_type=getattr(l, 'weak_type', False)) for l in leaves])
      got = (dataclasses.replace(exp, _get_vjp=None), box['out_def'])
      os.makedirs(self.xdir, exist_ok=True)
      tmp = f'{path}.{os.getpid()}.tmp'
      with open(tmp, 'wb') as f:
        f.write(pickle.dumps(got))
      os.replace(tmp, path)
    exp, out_def = got
    comp = jax.jit(exp.call).lower(*leaves).compile()
    fn_ = lambda ls: jax.tree.unflatten(out_def, comp(*ls))
    self._c[ck] = fn_
    return fn_(leaves)
