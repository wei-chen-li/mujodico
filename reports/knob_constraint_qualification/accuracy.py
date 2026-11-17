"""Check whether the inner contact solve supports its implicit derivatives."""
import argparse
import json
from types import SimpleNamespace
import jax
import jax.numpy as jp
import numpy as np
from run import ROOT,DATA,build,dump

def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--label',required=True)
  args=ap.parse_args();saved=json.loads((DATA/(args.label+'.json')).read_text())
  raw=np.load(DATA/(args.label+'.npz'));xs,ws=raw['xs'],raw['ws']
  results={};index=None
  for mode,tol in [('default',None),('tight',1e-12)]:
    cfg=SimpleNamespace(**saved['args']);cfg.inner_tolerance=tol
    task,tr,_,_,_,nw,_,_,_=build(cfg)
    ncf=nw-task.nu
    def force(y):return -tr.stage_equality(y[:tr.nx],jp.r_[y[tr.nx:],jp.zeros(ncf)])
    f=jax.jit(force)
    if index is None:
      predicted=np.asarray(jax.jit(jax.vmap(force))(np.hstack([xs[:-1],ws[:,:task.nu]])))
      index=int(np.unravel_index(np.argmax(abs(predicted-ws[:,task.nu:])),predicted.shape)[0])
    y=jp.r_[xs[index],ws[index,:task.nu]]
    supplied=np.asarray(jax.jit(jax.jacfwd(force))(y))
    checks=[]
    for eps in [1e-6,1e-8]:
      perturb=jp.eye(y.size)*eps
      yp=jax.jit(jax.vmap(force))(y[None]+perturb)
      ym=jax.jit(jax.vmap(force))(y[None]-perturb)
      fd=np.asarray((yp-ym)/(2*eps)).T
      norms=np.maximum(1,np.linalg.norm(fd,axis=1))
      checks.append({'epsilon':eps,'maximum_absolute_error':float(abs(fd-supplied).max()),
        'maximum_relative_row_error':float((np.linalg.norm(fd-supplied,axis=1)/norms).max())})
    results[mode]={'stage':index,'force':np.asarray(f(y)),'checks':checks}
  dump(ROOT/(args.label+'_accuracy.json'),results)
  print(json.dumps({k:v['checks'] for k,v in results.items()},indent=2))

if __name__=='__main__':main()
