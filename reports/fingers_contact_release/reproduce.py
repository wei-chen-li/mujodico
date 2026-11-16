"""Reproduce the selected continuation, optionally from the linear task start.

The default uses the supplied optimizer checkpoint. --from-linear regenerates
that checkpoint using the original task initialization and objective.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--from-linear',action='store_true')
  ap.add_argument('--prefix',default='repro')
  args=ap.parse_args()
  env=os.environ.copy()
  env.update(JAX_ENABLE_X64='True',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MPLCONFIGDIR='/tmp/mujodico-mpl')
  logs=ROOT/'logs';logs.mkdir(exist_ok=True)
  def phase(suffix,extra):
    label=args.prefix+'_'+suffix
    command=[sys.executable,str(ROOT/'run.py'),'--label',label,'--form','shooting',*extra]
    print('Running',label,'log:',logs/(label+'.log'),flush=True)
    with (logs/(label+'.log')).open('w') as log:
      subprocess.run(command,env=env,cwd=ROOT.parents[1],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((ROOT/'data'/(label+'.json')).read_text())
    print(label, 'status',result['status'],'objective',result['objective'],
          'constraint violation',result['constraint_violation'],flush=True)
    return label
  seed=str(ROOT/'data/checkpoints/frozen_ms_seed.npz')
  accurate=['--inner-tolerance','1e-12','--inner-iterations','100']
  finish=['--hessian','full',*accurate,'--warm-duals','--mu-init','1e-5']
  if args.from_linear:
    native=phase('native',['--smoothing','1e-5','--iterations','1000',
      '--inner-tolerance','1e-10','--inner-iterations','60'])
    polish=phase('polish',['--init',native,'--smoothing','1e-5','--iterations','373',*finish])
    seed=str(ROOT/'data/checkpoints'/(polish+'_best.npz'))
    if not Path(seed).exists():raise RuntimeError('No feasible optimizer checkpoint was generated; inspect the phase log.')
  c2=phase('c2',['--init',seed,'--smoothing','1e-5','--geometry-width','.001','--iterations','500',*finish])
  if json.loads((ROOT/'data'/(c2+'.json')).read_text())['status']!=0:
    raise RuntimeError('The smooth-model finishing phase did not converge; inspect its diagnostics.')
  physical=phase('physical',['--init',c2,'--smoothing','1e-7','--tau-factor','0',
      '--geometry-width','.001','--iterations','500',*finish])
  subprocess.run([sys.executable,str(ROOT/'validate.py'),'--label',physical],env=env,check=True)
  subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--best',physical,'--replay','zero'],env=env,check=True)


if __name__=='__main__':main()
