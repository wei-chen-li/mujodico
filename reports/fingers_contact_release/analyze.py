"""Compare identical-cost finger experiments and render regrasp trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)


def load_results():
  results={}
  for p in DATA.glob('*.json'):
    if p.name.endswith('_history.json'):continue
    s=json.loads(p.read_text())
    if 'args' in s:results[p.stem]=s
  return results


def segments(x):
  result=[]
  for base,i in [(np.array([-.15,0.]),0),(np.array([.15,0.]),2)]:
    q1,q2=x[i],x[i+1]
    d1=np.array([-np.sin(q1),np.cos(q1)])
    d2=np.array([-np.sin(q1+q2),np.cos(q1+q2)])
    joint=base+.15*d1
    result.append((base,joint,joint+.06*d2,joint+.15*d2))
  return result


def material_events(xs,gaps,mask,side):
  from run import release_events
  g=np.min(gaps[:,mask],axis=1)
  events=release_events(g)
  side=0 if side=='left' else 1
  for e in events:
    before=int(np.flatnonzero(g[:e['release']]<=.0005)[-1])
    angles=[]
    for k in [before,e['recontact']]:
      center=xs[k,4:6];seg=segments(xs[k])[side]
      pairs=[(seg[0],seg[1]),(seg[2],seg[3])]
      points=[]
      for a,b in pairs:
        t=np.clip((center-a)@(b-a)/((b-a)@(b-a)),0,1)
        points.append(a+t*(b-a))
      p=min(points,key=lambda p:np.linalg.norm(p-center))
      angles.append(np.arctan2(p[1]-center[1],p[0]-center[0])-xs[k,6])
    delta=(angles[1]-angles[0]+np.pi)%(2*np.pi)-np.pi
    e.update(last_contact=before,release_seconds=e['release']*.02,recontact_seconds=e['recontact']*.02,
             max_gap_mm=float(np.max(g[e['release']:e['recontact']])*1000),
             material_contact_change_degrees=float(np.degrees(delta)))
  return events


def animation(labels,results,replay):
  width,height=450,460
  items=[]
  for label in labels:
    raw=np.load(DATA/(label+'.npz'))
    items.append((label,raw[replay+'_xs'],raw[replay+'_gaps'],results[label]))
  frames=[]
  for k in range(0,101,2):
    frame=Image.new('RGB',(width*len(items),height),'white');draw=ImageDraw.Draw(frame)
    for col,(label,xs,gaps,s) in enumerate(items):
      offset=col*width
      def point(p):return(offset+int((p[0]+.35)/.7*width),height-45-int((p[1]+.04)/.59*(height-100)))
      def disk(p,r,color):
        c=point(p);rx=int(r/.7*width);ry=int(r/.59*(height-100))
        draw.ellipse((c[0]-rx,c[1]-ry,c[0]+rx,c[1]+ry),fill=color)
      def capsule(a,b,r,color):
        d=b-a;n=np.array([-d[1],d[0]])/np.linalg.norm(d)*r
        draw.polygon([point(a+n),point(b+n),point(b-n),point(a-n)],fill=color)
        disk(a,r,color);disk(b,r,color)
      x=xs[k]
      draw.text((offset+12,10),label,fill='black')
      draw.text((offset+12,28),f't = {k*.02:.2f} s, angle = {np.degrees(x[6]):.1f} deg',fill='black')
      draw.text((offset+12,44),f'{replay}-smoothing replay, objective = {s["replay"][replay]["objective"]:.4f}',fill='black')
      for side,seg,color in zip(['left','right'],segments(x),['#377cb5','#e28a34']):
        capsule(seg[0],seg[1],.03,color);capsule(seg[2],seg[3],.03,color)
        draw.line([point(seg[1]),point(seg[3])],fill=color,width=2)
        disk(seg[0],.008,'#555555');disk(seg[1],.007,'#555555')
        mask=np.asarray(s['finger_contact_masks'][side],bool)
        gap=float(np.min(gaps[k,mask]))
        draw.text((offset+12,405 if side=='left' else 425),f'{side}: gap {gap*1000:.1f} mm',fill=color)
      disk(x[4:6],.08,'#535c68')
      p=x[4:6]+.055*np.array([np.cos(x[6]),np.sin(x[6])])
      disk(p,.01,'#ffffff')
    frames.append(frame)
  frames[0].save(FIG/'comparison.gif',save_all=True,append_images=frames[1:],duration=40,loop=0)


def main():
  ap=argparse.ArgumentParser(__doc__);ap.add_argument('--best');ap.add_argument('--baseline',default='onlysphere_ippas')
  ap.add_argument('--replay',choices=['smooth','zero'],default='zero')
  args=ap.parse_args();results=load_results();rows=[]
  for label,s in sorted(results.items()):
    row=dict(label=label,pairs=len(s['contact_pairs']),form=s['args']['form'],solver=s['args'].get('solver','ipopt'),
      start=s['args'].get('init') or 'linear',smoothing=s['args']['smoothing'],hessian=s['args']['hessian'],
      force_floor=s['args']['force_floor'],status=s['status'],iterations=s['iterations'],objective=s['objective'],
      constraint_violation=s['constraint_violation'],force_defect=s['force_equality_defect'],
      angle_error_degrees=np.degrees(s['angle_error_radians']),position_error_mm=s['position_error_m']*1000,
      smooth_replay_objective=s['replay']['smooth']['objective'],zero_replay_objective=s['replay']['zero']['objective'],
      smooth_replay_angle_error_degrees=np.degrees(s['replay']['smooth']['angle_error_radians']),
      zero_replay_angle_error_degrees=np.degrees(s['replay']['zero']['angle_error_radians']),
      smooth_replay_state_difference=s['replay']['smooth']['max_state_difference'],
      left_regrasps=len(s['replay']['smooth']['contacts']['left']['regrasp_events']),
      right_regrasps=len(s['replay']['smooth']['contacts']['right']['regrasp_events']))
    rows.append(row)
  with (ROOT/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  print('Saved',len(rows),'results.')
  if all((DATA/(n+'_history.json')).exists() for n in ['paired_plain','ms_c2geo']):
    fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for label in ['paired_plain','ms_c2geo']:
      h=json.loads((DATA/(label+'_history.json')).read_text());t=[v['iter'] for v in h]
      for ax,key in zip(axs.flat,['objective','g_violation','grad_lag_x','mu']):
        values=[v.get('unscaled',{}).get(key,v.get(key)) for v in h]
        ax.plot(t,values,label=label)
        ax.set_ylabel(key);ax.set_xlabel('Additional IPOPT iterations');ax.grid(alpha=.2)
        if key!='objective':ax.set_yscale('log')
        ax.legend()
    for ext in ['png','pdf']:fig.savefig(FIG/('geometry_finishing.'+ext),dpi=180)
    plt.close(fig)
  if not args.best:return
  labels=[args.baseline,args.best]
  fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
  events={}
  for label in labels:
    s=results[label];raw=np.load(DATA/(label+'.npz'));x=raw[args.replay+'_xs'];g=raw[args.replay+'_gaps'];t=np.arange(len(x))*.02
    axes[0].plot(t,np.degrees(x[:,6]),label=label)
    for ax,side in zip(axes[1:],['left','right']):
      mask=np.asarray(s['finger_contact_masks'][side],bool)
      ax.plot(t,1000*np.min(g[:,mask],axis=1),label=label)
      ax.set_ylabel(side+' minimum gap (mm)')
      events[label+'_'+side]=material_events(x,g,mask,side)
      ax.axhline(.5,color='k',ls=':',alpha=.3)
      ax.axhline(2.,color='k',ls='--',alpha=.3)
  axes[0].axhline(90,color='k',ls=':',label='Original goal')
  axes[0].set_ylabel('Replayed object angle (degrees)')
  for ax in axes:
    ax.grid(alpha=.2);ax.legend()
  axes[-1].set_xlabel('Time (s)')
  for ext in ['png','pdf']:fig.savefig(FIG/('release_comparison.'+ext),dpi=180,bbox_inches='tight')
  (ROOT/'contact_events.json').write_text(json.dumps(events,indent=2))
  animation(labels,results,args.replay)


if __name__=='__main__':main()
