import sys, time; sys.path.insert(0,'mujodico/docs/handoff_2026-11-23_bench/scripts')
import numpy as np, mujoco, embodiments as E, handik as H
from mujodico.examples.tasks.task import Task
from simtools import sim_true, keyframes
q0=np.array([0,1,0,1])/np.sqrt(2); qd=E.quat_mul(E.axis_quat([1,0,0],np.pi), q0)
env='''    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" contype="2" conaffinity="1"/>
    <body name="tool" pos="0.05 0 0.015"><freejoint name="tool"/><geom name="handle" type="capsule" size="0.015" fromto="0 -0.08 0 0 0.08 0" contype="1" conaffinity="2" density="500" friction="1"/></body>'''
m=E.allegro_model(env, wrist_pos=[0,0,0], palm_quat=qd)
W=7; xh,zh,rh,rt=0.05,0.015,0.015,0.012
def grasp_q(inward):
    q=np.zeros(m.nq); q[0:3]=[0.05,0,0.015]; q[3]=1; q[W:W+6]=[0.04,-0.013,0.119,0,0,0]; q[W+6+12]=1.4
    s=rh+rt-0.002-inward
    tg={H.tip_geom(m,'ff'):[xh+s,0.04,zh+0.004],H.tip_geom(m,'mf'):[xh+s,0,zh+0.004],H.tip_geom(m,'rf'):[xh+s,-0.04,zh+0.004],H.tip_geom(m,'th'):[xh-s,0,zh+0.004]}
    qs,e=H.ik(m,q,tg,sum(H.FINGER_JOINTS.values(),[]),iters=3000); return qs[W+6:W+22],e
qc,e1=grasp_q(0.0); qsq,e2=grasp_q(0.015); qo,e3=grasp_q(-0.03)
print('ik errs',e1,e2,e3); print('open',np.round(qo,2)); print('squeeze',np.round(qsq,2))
x0=np.zeros(m.nq+m.nv); x0[0:3]=[0.05,0,0.015-1e-4]; x0[3]=1; x0[W:W+6]=[0.04,-0.013,0.20,0,0,0]; x0[W+6:W+22]=qo
nx=m.nq+m.nv; Q=np.zeros((nx,nx)); Q[0,0]=Q[1,1]=Q[2,2]=100
uref=np.r_[x0[W:W+6],qo]
class Lift(Task): pass
t=Lift(model=m,horizon=100,x_0=x0,x_goal=x0.copy(),Q=Q,Qf=Q,u_ref=uref,R=np.eye(22)*0.01,contact_smoothing=0.002)
names=[m.actuator(i).name for i in range(m.nu)]
fa=[n for n in names if not n.startswith('a_')]
def F(qf): return {n:v for n,v in zip(fa,qf)}
kf=[(0.8,dict(a_wz=0.119)),(1.0,{}),(1.6,F(qsq)),(2.0,{}),(3.0,dict(a_wz=0.25)),(4.0,dict(a_wrx=1.0,a_wx=0.0))]
us=keyframes(t,kf)
tic=time.time(); X=sim_true(t,us); print('true law', time.time()-tic)
for k in range(0,101,10): print(k,'tool',np.round(X[k,0:7],3),'wrist',np.round(X[k,W:W+6],3))
