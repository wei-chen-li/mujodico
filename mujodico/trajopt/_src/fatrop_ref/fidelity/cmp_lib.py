"""Build the same OCP in CasADi(FATROP) and IpPasSolver and compare traces."""
import re, sys, io, os, numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import casadi as ca
from mujodico.trajopt._src.fatrop_ref.ippas import IpPasSolver, IpPasOptions

def run_casadi(N,nx,nw,x0,dyn_ca,cost_ca,term_ca,eq_ca,ineq_ca,lb,ub,xs0,ws0,max_iter,opts=None):
    X=[ca.MX.sym(f'x{k}',nx) for k in range(N+1)]; U=[ca.MX.sym(f'u{k}',nw) for k in range(N)]
    v=[];g=[];lbg=[];ubg=[];eqf=[];J=0;x_init=[]
    for k in range(N):
        v+=[X[k],U[k]]; x_init+=[xs0[k],ws0[k]]
        if k==0: g.append(X[0]-ca.DM(x0)); lbg+=[0]*nx; ubg+=[0]*nx; eqf+=[True]*nx
        g.append(X[k+1]-dyn_ca(X[k],U[k])); lbg+=[0]*nx; ubg+=[0]*nx; eqf+=[True]*nx
        e=eq_ca(X[k],U[k])
        if e is not None and e.shape[0]: g.append(e); lbg+=[0]*e.shape[0]; ubg+=[0]*e.shape[0]; eqf+=[True]*e.shape[0]
        gi=ineq_ca(X[k],U[k]); g.append(gi); lbg+=list(lb); ubg+=list(ub); eqf+=[False]*len(lb)
        J+=cost_ca(X[k],U[k])
    v.append(X[N]); x_init.append(xs0[N]); J+=term_ca(X[N])
    o={'structure_detection':'auto','equality':eqf,'fatrop.print_level':10,'fatrop.max_iter':max_iter}
    if opts: o.update({'fatrop.'+k:val for k,val in opts.items()})
    S=ca.nlpsol('s','fatrop',{'x':ca.vertcat(*v),'f':J,'g':ca.vertcat(*g)},o)
    # capture C-level stdout
    r,wfd=os.pipe(); saved=os.dup(1); os.dup2(wfd,1)
    try:
        res=S(x0=np.concatenate([np.ravel(a) for a in x_init]),lbg=lbg,ubg=ubg)
    finally:
        sys.stdout.flush(); os.dup2(saved,1); os.close(wfd)
    txt=os.read(r,10**8).decode(errors='replace'); os.close(r)
    rows=[]
    for line in txt.splitlines():
        m=re.match(r'\s*(\d+)(r?)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\d]+)\s+([-+.\deE]+)\s+(\S+)\s+([-+.\deE]+)\s+([-+.\deE]+)(\S)\s+(\d+)',line)
        if m: rows.append(m.groups())
    xv=np.asarray(res['x']).ravel()
    return S.stats()['return_status'], rows, xv, txt

def compare(rows_c, trace_j, n=None):
    from mujodico.trajopt._src.fatrop_ref.ippas import TYPE_CHARS
    out=[]; first_div=None
    m=min(len(rows_c),len(trace_j)) if n is None else n
    for i in range(m):
        c=rows_c[i]; j=trace_j[i]
        cobj=float(c[2]); jobj=j[2]
        same=(c[0]==str(int(j[0])) and (c[1]=='r')==bool(j[1]) and abs(cobj-jobj)<=1e-7*max(1,abs(cobj)) and c[10]==TYPE_CHARS[int(j[10])])
        if not same and first_div is None: first_div=i
        out.append(f"{'  ' if same else '!!'} C: {c[0]:>3}{c[1]:1} {c[2]:>15} {c[3]:>9} {c[4]:>9} {c[5]:>5} {c[8]:>9} {c[9]:>9}{c[10]} {c[11]:>2}  J: {int(j[0]):3d}{'r' if j[1] else ' '} {j[2]:15.8e} {j[3]:9.2e} {j[4]:9.2e} {j[5]:5.1f} {j[8]:9.2e} {j[9]:9.2e}{TYPE_CHARS[int(j[10])]} {int(j[11]):2d}")
    return first_div, '\n'.join(out)
