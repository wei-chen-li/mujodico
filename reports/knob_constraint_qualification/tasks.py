"""Controlled Cartesian, one-finger, and two-finger knob models."""
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from mujodico.examples.tasks import PushCircularKnob,FingerCircularKnob
from mujodico.examples.tasks.finger_circular_knob import _XML
from mujodico.examples.tasks.task import Task

def make_task(name,*,tip_only=False,friction=None,goal_degrees=360,finger_collisions=False,mirror_axis='y',same_joint_angles=False):
  if name in ['push','quadrant']:
    task=PushCircularKnob()
    task.x_0[2]=0.;task.x_goal[2]=np.deg2rad(goal_degrees)
    if name=='quadrant':task.model.jnt_range[:2,1]=0.
  elif name=='finger':
    task=FingerCircularKnob();task.x_goal[2]=np.deg2rad(goal_degrees)
  else:
    root=ET.fromstring(_XML)
    world=root.find('worldbody');first=world.find('body')
    second=copy.deepcopy(first)
    if mirror_axis=='x':
      second.set('pos','-0.15 0.25 0');second.set('quat','0 1 0 0')
    else:second.set('pos','0.15 -0.25 0')
    for node in second.iter():
      if 'name' in node.attrib:node.set('name',node.get('name').replace('finger','right_finger'))
    world.insert(1,second)
    actuator=root.find('actuator')
    for node in list(actuator):
      item=copy.deepcopy(node)
      item.set('name',item.get('name').replace('finger','right_finger'))
      item.set('joint',item.get('joint').replace('finger','right_finger'))
      actuator.append(item)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    angles=[30,-60,30,-60] if mirror_axis=='x' or same_joint_angles else [30,-60,-30,60]
    x0=np.r_[np.deg2rad(angles),0.,np.zeros(5)]
    goal=np.r_[np.zeros(4),np.deg2rad(goal_degrees),np.zeros(5)]
    Q=np.diag([0.,0.,0.,0.,1.,0.,0.,0.,0.,.001])
    task=Task(model,400,x0,goal,Q,Q*400,np.zeros(4),np.eye(4)*.01,.01)
  model=task.model
  # Only finger/knob interactions; extra finger-finger contacts are a confound.
  if name not in ['push','quadrant']:
    model.geom_contype[:]=1;model.geom_conaffinity[:]=2
    knob=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'knob')
    sphere=model.body_geomadr[knob]
    model.geom_contype[sphere]=2;model.geom_conaffinity[sphere]=1
    model.geom_contype[-1]=model.geom_conaffinity[-1]=0
    if tip_only:
      for i in range(model.ngeom):
        body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i]))
        if body.endswith('link1'):model.geom_contype[i]=model.geom_conaffinity[i]=0
    if finger_collisions:
      model.geom_contype[:-1]=model.geom_conaffinity[:-1]=1
  if friction is not None:model.geom_friction[:,0]=friction
  return task
