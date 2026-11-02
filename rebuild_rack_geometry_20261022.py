"""Regenerate tighter rack collision pieces from the released collision mesh.
Visual geometry and public mass/inertia are preserved.
"""
import argparse,json,copy,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
from improvement_common import BASE,save,utc,digest
from batch_native import _docker
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare-cache-only',action='store_true');a=p.parse_args();r=a.run;cfg=json.loads((r/'episode-config.json').read_text());assert cfg['episode']==30 and not (r/'coherent-input-checksums.json').exists();meshdir=r/'reconstruction/mesh_priors';urdf=meshdir/'plastic_dish_rack.urdf';cache=r/'research/rebuilt-rack-convex-hulls.npz';report=r/'research/rebuilt-rack-geometry.json'
if not cache.exists():
 code="""from pathlib import Path
import json,numpy as np,trimesh,xml.etree.ElementTree as ET
from flash_chord.assets.mesh import convex_decompose
r=Path('/episode');u=ET.parse(r/'reconstruction/mesh_priors/plastic_dish_rack.urdf');path=Path(u.find('.//collision/geometry/mesh').get('filename'));m=trimesh.load(path,force='mesh',process=False);hulls=convex_decompose(m.vertices,m.faces,threshold=.01);arrays={}
for i,h in enumerate(hulls):arrays[f'v{i}']=h.vertices;arrays[f'f{i}']=h.faces
np.savez_compressed(r/'research/rebuilt-rack-convex-hulls.npz',**arrays);print('New rack hull count',len(hulls),flush=True)
"""
 (r/'scripts/rebuild_rack_convex_cache.py').write_text(code);image=_docker(['inspect','v2d-batch-episode30-newton','--format','{{.Image}}']).stdout.strip();_docker(['run','--rm','--ipc','host','-e','CUDA_VISIBLE_DEVICES=','-e','JAX_PLATFORMS=cpu','-e','OMP_NUM_THREADS=4','-v',f'{r.resolve()}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/flash_chord:/workspace','--workdir','/workspace','--entrypoint','python',image,'-u','/episode/scripts/rebuild_rack_convex_cache.py'],r/'logs/rebuild-rack-convex-cache.log')
if a.prepare_cache_only:raise SystemExit(0)
if report.exists() and json.loads(report.read_text()).get('urdf_sha256')==digest(urdf):raise SystemExit(0)
tree=ET.parse(urdf);link=tree.find('.//link');collisions=list(link.findall('collision'));assert len(collisions)==1;original=collisions[0];visual=ET.tostring(link.find('visual'));inertial=ET.tostring(link.find('inertial'));link.remove(original);arrays=np.load(cache);parts=[]
for i in range(len(arrays.files)//2):
 path=meshdir/f'plastic_dish_rack_fine_{i:03d}.obj';trimesh.Trimesh(arrays[f'v{i}'],arrays[f'f{i}'],process=False).export(path);node=copy.deepcopy(original);node.find('geometry/mesh').set('filename','/episode/reconstruction/mesh_priors/'+path.name);link.append(node);parts.append({'path':path.name,'sha256':digest(path)})
assert ET.tostring(link.find('visual'))==visual and ET.tostring(link.find('inertial'))==inertial;tree.write(urdf);save(report,{'updated_utc':utc(),'coacd_threshold':.01,'hull_count':len(parts),'visual_mesh_unchanged':True,'public_mass_and_inertia_unchanged':True,'urdf_sha256':digest(urdf),'collision_parts':parts});print('Fresh rack collision pieces applied',len(parts),flush=True)
