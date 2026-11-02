"""Remove internal axial seams from the same fitted hollow collision shell.

Each angular wall is a linear frustum split into four adjacent height intervals.
Merge those intervals only after checking that their union is already convex.
This is a separate diagnostic scene; running learners and source meshes stay frozen.
"""
import copy,hashlib,json,shutil,stat,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
from improvement_common import BASE,save,utc
from batch_native import _docker
SOURCE=BASE/'runs/episode_000010_newton_stable_contact_20261020'
R=BASE/'runs/episode_000010_newton_continuous_shell_probe_20261020'
assert not R.exists();R.mkdir()
for name in ['logs','research','evaluation','videos','deliverables']:(R/name).mkdir()
shutil.copytree(SOURCE/'scripts',R/'scripts')
(R/'reconstruction').mkdir()
for p in (SOURCE/'reconstruction').iterdir():
 if p.name=='mesh_priors':shutil.copytree(p,R/'reconstruction'/p.name)
 else:(R/'reconstruction'/p.name).symlink_to(p,target_is_directory=p.is_dir())
for name in ['track3_hollow_supported_candidate','track3_collision_variant','track3_video']:
 shutil.copytree(SOURCE/'human_motion_data'/name,R/'human_motion_data'/name)
shutil.copytree(SOURCE/'research/initial-policy',R/'research/initial-policy')
cfg=json.loads((SOURCE/'episode-config.json').read_text());cfg.update(run=str(R),native_container='v2d-shell-probe-episode10-newton',training_variant='continuous_shell_probe_20261020',predecomposed_object_collisions=True)
save(R/'episode-config.json',cfg)
(R/'.venv-video').symlink_to(SOURCE/'.venv-video',target_is_directory=True)
geometry=R/'reconstruction/mesh_priors';report={}
for name in cfg['objects']:
 p=geometry/f'{name}.urdf';tree=ET.parse(p);link=tree.find('.//link');old=list(link.findall('collision'));assert len(old)==193
 groups={};base=[]
 for node in old:
  file=Path(node.find('geometry/mesh').get('filename')).name
  if file.endswith('_base.obj'):base.append(copy.deepcopy(node));continue
  sector=int(Path(file).stem.split('_')[-1]);groups.setdefault(sector,[]).append((node,geometry/file))
 assert len(groups)==48 and len(base)==1
 volumes=[];errors=[]
 for sector,parts in sorted(groups.items()):
  assert len(parts)==4
  meshes=[trimesh.load(file,force='mesh',process=False) for _,file in parts]
  hull=trimesh.convex.convex_hull(np.concatenate([m.vertices for m in meshes]));before=sum(abs(m.volume) for m in meshes)
  relative_error=abs(abs(hull.volume)-before)/before
  assert relative_error<2e-5,(name,sector,relative_error)
  # All exterior support planes must also support the component vertices.
  from scipy.spatial import ConvexHull
  planes=ConvexHull(hull.vertices).equations
  vertices=np.concatenate([m.vertices for m in meshes]);excess=np.max(vertices@planes[:,:3].T+planes[:,3])
  assert excess<1e-7,(name,sector,excess)
  filename=f'{name}_continuous_shell_{sector:02d}.obj';hull.export(geometry/filename)
  node=copy.deepcopy(parts[0][0]);node.find('geometry/mesh').set('filename','/episode/reconstruction/mesh_priors/'+filename)
  volumes.append(relative_error);errors.append(float(excess));groups[sector]=node
 for node in old:link.remove(node)
 for node in [*groups.values(),*base]:link.append(node)
 original=ET.parse(SOURCE/'reconstruction/mesh_priors'/p.name).find('.//link')
 assert ET.tostring(original.find('inertial'))==ET.tostring(link.find('inertial'))
 assert ET.tostring(original.find('visual'))==ET.tostring(link.find('visual'))
 tree.write(p,encoding='unicode')
 report[name]={'previous_collision_parts':193,'continuous_collision_parts':49,
  'maximum_relative_union_volume_error':max(volumes),'maximum_component_vertex_support_error_m':max(errors),
  'visual_mass_and_inertia_identical':True,'external_hollow_frustum_dimensions_unchanged':True}
save(R/'research/continuous-shell-geometry-contract.json',{'updated_utc':utc(),'source_run':str(SOURCE),'diagnostic_only':True,
 'approved_for_training':False,'objects':report,'source_camera_reference_and_clock_unchanged':True})
info=json.loads(_docker(['inspect',cfg['native_container'].replace('shell-probe','stable')]).stdout)[0]
args=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',cfg['native_container']]
for mount in info['Mounts']:
 source=str(R) if mount['Destination']=='/episode' else mount['Source']
 args+=['-v',source+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
for node in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
 if stat.S_ISCHR(node.stat().st_mode):args+=['--device',str(node)]
args+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','--workdir','/workspace','--entrypoint','/bin/sleep',info['Image'],'infinity']
_docker(args,R/'logs/create-newton.log')
for p,h in json.loads((SOURCE/'coherent-input-checksums.json').read_text()).items():assert hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()==h,p
print(json.dumps(report),flush=True)
