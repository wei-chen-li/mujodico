"""Locate remaining stationarity and check one-sided Lagrangian derivatives."""
import argparse
import json
import numpy as np
from scipy import sparse
from convergence_diagnostics import ROOT,OUT,build,dump,describe_variable


def main():
  ap=argparse.ArgumentParser(__doc__)
  ap.add_argument('--source',required=True)
  ap.add_argument('--checkpoint',required=True)
  ap.add_argument('--check-hessian',action='store_true')
  args=ap.parse_args()
  saved=json.loads((OUT/(args.checkpoint+'.json')).read_text())
  cfg=saved['args']
  _,task,trans,nlp,_=build(args.source,exact=args.check_hessian,inner_tolerance=cfg.get('inner_tolerance'),
                         inner_iterations=cfg.get('inner_iterations'))
  raw=np.load(OUT/(args.checkpoint+'.npz'))
  z=nlp._pack(raw['xs'],raw['ws'])
  nu=np.asarray(saved['info']['mult_g'])
  zl,zu=np.asarray(saved['info']['mult_x_L']),np.asarray(saved['info']['mult_x_U'])
  ij=np.asarray(nlp.jacobian_structure())
  def grad_at(z):
    J=sparse.coo_matrix((nlp.jacobian_entries(z),(ij[:,0],ij[:,1])),shape=(nlp.num_constraints,nlp.num_vars)).tocsr()
    return np.asarray(nlp.gradient(z))+J.T@nu-zl+zu
  rd=grad_at(z)
  c0=np.asarray(nlp.constraints(z))
  obj0=nlp.objective(z)
  top=[]
  for j in np.argsort(-np.abs(rd))[:12]:
    d=describe_variable(j,task,trans);d.update(residual=rd[j],primal_value=z[j]);top.append(d)
  qids=[j for j in range(len(z)) if j%(trans.nx+trans.nw)<task.model.nq]
  selected=sorted(qids,key=lambda j:-abs(rd[j]))[:3]
  H=None
  if args.check_hessian:
    hij=np.asarray(nlp.hessian_structure())
    lower=sparse.coo_matrix((nlp.hessian_entries(z,nu,1.),(hij[:,0],hij[:,1])),shape=(len(z),len(z))).tocsr()
    H=lower+lower.T-sparse.diags(lower.diagonal())
  checks=[]
  for j in selected:
    d=describe_variable(j,task,trans);d['analytic_lagrangian_gradient']=rd[j]
    one_sided=[]
    for h in [1e-4,1e-6,1e-8,1e-10]:
      zp,zm=z.copy(),z.copy();zp[j]+=h;zm[j]-=h
      dp=(nlp.objective(zp)-obj0+nu@(np.asarray(nlp.constraints(zp))-c0))/h-zl[j]+zu[j]
      dm=(obj0-nlp.objective(zm)+nu@(c0-np.asarray(nlp.constraints(zm))))/h-zl[j]+zu[j]
      gp,gm=grad_at(zp),grad_at(zm)
      entry={'step':h,'positive_difference':dp,'negative_difference':dm,
             'gradient_positive':gp[j],'gradient_negative':gm[j]}
      if H is not None:
        hv=H[:,j].toarray().ravel();fd=(gp-gm)/(2*h)
        entry['hessian_column_max_error']=float(np.max(np.abs(fd-hv)))
        entry['hessian_column_relative_error']=float(np.linalg.norm(fd-hv)/max(1,np.linalg.norm(hv)))
      one_sided.append(entry)
    d['one_sided']=one_sided
    k=d['stage'];x=raw['xs'][k]
    if task.model.nq==5:
      rel=x[:2]-x[2:4];theta=x[4];half=np.array([.15,.08])
    else:
      rel=x[:2];theta=x[2];half=np.array([.08,.08])
    local=np.array([np.cos(theta)*rel[0]+np.sin(theta)*rel[1],-np.sin(theta)*rel[0]+np.cos(theta)*rel[1]])
    d['pusher_in_box_frame']=local
    d['distance_to_projection_feature_boundary']=np.abs(np.abs(local)-half)
    checks.append(d)
  result={'source':args.source,'checkpoint':args.checkpoint,
          'max_stationarity':float(np.max(np.abs(rd))),'top_stationarity':top,
          'worst_position_derivative_checks':checks}
  first=checks[0]
  if task.model.nq==3 and first['component']=='pusher_y':
    # Hold the multipliers and all other decision coordinates fixed, move the
    # selected state exactly to its nearest x-projection boundary, and compare
    # limiting gradients from the face and corner sides.
    k=first['stage'];x=raw['xs'][k];local=np.asarray(first['pusher_in_box_frame'])
    target=np.sign(local[0])*.08
    j=first['index'];zb=z.copy();zb[j]+=(target-local[0])/np.sin(x[2])
    values=[]
    for h in [1e-6,1e-8,1e-10]:
      zp,zm=zb.copy(),zb.copy();zp[j]+=h;zm[j]-=h
      values.append({'step':h,'gradient_positive':grad_at(zp)[j],'gradient_negative':grad_at(zm)[j]})
    result['closest_feature_side_gradients']={'stage':k,'variable':first['component'],
       'distance_moved_to_boundary':float(zb[j]-z[j]),'side_gradients':values}
  dump(OUT/(args.checkpoint+'_stationarity.json'),result)
  print(json.dumps(result,default=lambda x:np.asarray(x).tolist()),flush=True)


if __name__=='__main__':main()
