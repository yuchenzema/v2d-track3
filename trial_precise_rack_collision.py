"""Test tighter convex decomposition of the released rack without changing its visual mesh."""
from pathlib import Path
import copy,fcntl,hashlib,json,shutil,time,xml.etree.ElementTree as ET
import numpy as np,trimesh
from batch_native import containers,docker
BASE=Path('/home/yu_eli/v2d-track3');r=BASE/'runs/episode_000015_newton_batch_20261015'
with (r/'research/collision-reference.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists()
 meshdir=r/'reconstruction/mesh_priors';urdf=meshdir/'plastic_dish_rack.urdf';backup=r/f'research/rack-collision-trial-{time.time_ns()}';backup.mkdir()
 for p in [urdf,r/'evaluation/object-collision-hulls.npz',r/'evaluation/track3_collision_variant-collision-audit.json']:
  shutil.copy2(p,backup/p.name)
 arrays=np.load(r/'research/rack-precise-convex-hulls.npz');tree=ET.parse(urdf);link=tree.getroot().find('link');original=link.find('collision');link.remove(original);parts=[]
 for i in range(len(arrays.files)//2):
  name=f'plastic_dish_rack_precise_{i:03d}.obj';path=meshdir/name;trimesh.Trimesh(arrays[f'v{i}'],arrays[f'f{i}'],process=False).export(path)
  node=copy.deepcopy(original);node.find('geometry/mesh').set('filename','/episode/reconstruction/mesh_priors/'+name);link.append(node);parts.append({'mesh':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 tree.write(urdf);newton,_=containers(r,15);shutil.copy2(BASE/'native_reference_audit.py',r/'scripts/native_reference_audit.py')
 try:
  docker(['exec',newton,'env','V2D_AUDIT_REFERENCE=track3_collision_variant','python','-u','/episode/scripts/native_reference_audit.py'],r/'logs/rack-precise-collision-audit.log')
  before=json.loads((backup/'track3_collision_variant-collision-audit.json').read_text());after=json.loads((r/'evaluation/track3_collision_variant-collision-audit.json').read_text())
  def penalty(a):return sum(sum(max(0,x['penetration_mm']-3)**2 for x in f['penetrating_cross_body_contacts']) for f in a['frames'])
  accepted=penalty(after)<penalty(before)
  report={'method':__doc__,'threshold':.01,'hull_count':len(parts),'accepted':accepted,'old_penalty':penalty(before),'new_penalty':penalty(after),'old_frames_over_10mm':before['source_frames_over_10mm'],'new_frames_over_10mm':after['source_frames_over_10mm'],'backup':str(backup.relative_to(r)),'parts':parts,'released_visual_mesh_unchanged':True,'mass_and_inertia_unchanged':True}
  if not accepted:
   for p in [urdf,r/'evaluation/object-collision-hulls.npz',r/'evaluation/track3_collision_variant-collision-audit.json']:shutil.copy2(backup/p.name,p)
  (r/'research/precise-rack-collision.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='parts'},flush=True)
 except BaseException:
  for p in [urdf,r/'evaluation/object-collision-hulls.npz',r/'evaluation/track3_collision_variant-collision-audit.json']:shutil.copy2(backup/p.name,p)
  raise
