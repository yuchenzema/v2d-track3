"""Build tighter cup collision hulls from unchanged released visual meshes."""
from pathlib import Path
import copy,hashlib,json,xml.etree.ElementTree as ET
import numpy as np,trimesh
from flash_chord.assets.mesh import convex_decompose
r=Path('/episode');m=r/'reconstruction/mesh_priors';results={}
assert not (r/'coherent-input-checksums.json').exists()
for name in ['blue_cup','beige_cup']:
 meshfile=m/f'{name}.obj';mesh=trimesh.load(meshfile,force='mesh',process=False)
 hulls=convex_decompose(np.asarray(mesh.vertices),np.asarray(mesh.faces),threshold=.01,
  cache_dir=r/'research/precise-cup-cache')
 p=m/f'{name}.urdf';backup=r/'research'/f'{name}-collision-before-fine.urdf'
 if not backup.exists():backup.write_bytes(p.read_bytes())
 tree=ET.parse(backup);link=tree.find('.//link');template=copy.deepcopy(link.find('collision'))
 for node in link.findall('collision'):link.remove(node)
 for i,hull in enumerate(hulls):
  part=m/f'{name}_precise_{i:03d}.obj';trimesh.Trimesh(hull.vertices,hull.faces,process=False).export(part)
  node=copy.deepcopy(template);node.find('geometry/mesh').set('filename','/episode/reconstruction/mesh_priors/'+part.name);link.append(node)
 tree.write(p,encoding='unicode')
 results[name]={'threshold':.01,'collision_pieces':len(hulls),'visual_sha256':hashlib.sha256(meshfile.read_bytes()).hexdigest(),
  'visual_mesh_mass_inertia_unchanged':True,'collision_urdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(name,results[name],flush=True)
(r/'research/fine-cup-collision-candidate.json').write_text(json.dumps({'diagnostic_only':True,'approved_for_training':False,'objects':results},indent=2))
