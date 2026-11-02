"""Fit thin hollow cup collision shells to released scans and observed deep nesting.

The released visual meshes are unchanged. Wall thickness is a declared 1 mm
reconstruction assumption, checked subsequently against source pixels and gravity.
"""
from pathlib import Path
import copy,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,trimesh
from scipy.optimize import least_squares
r=Path('/episode');root=r/'reconstruction/mesh_priors';assert not (r/'coherent-input-checksums.json').exists();report={}
for name in ['blue_cup','beige_cup']:
 file=root/f'{name}.obj';mesh=trimesh.load(file,force='mesh',process=False);v=np.asarray(mesh.vertices);fitv=v[abs(v[:,2])<.06];z=-fitv[:,2]
 def residual(x):return np.linalg.norm(fitv[:,:2]-x[:2]-z[:,None]*x[2:4],axis=1)-(x[4]+x[5]*z)
 fit=least_squares(residual,[0.,0.,0.,0.,.04,.06],loss='soft_l1',f_scale=.001)
 cx,cy,sx,sy,radius,slope=fit.x;radius+=np.percentile(residual(fit.x),85)
 axis=np.array([sx,sy,-1.]);axis/=np.linalg.norm(axis);u=np.array([1.,0.,sx]);u-=axis*np.dot(u,axis);u/=np.linalg.norm(u);w=np.cross(axis,u);origin=np.array([cx,cy,0.])
 height=(v-origin)@axis;bottom,top=np.percentile(height,[.5,99.5]);knots=np.linspace(bottom,top,5);wall=.001;segments=48
 assert .03<radius<.05 and .025<slope<.1 and .14<top-bottom<.19
 urdf=root/f'{name}.urdf';backup=r/'research'/f'{name}-fine-hulls-before-hollow.urdf'
 if not backup.exists():backup.write_bytes(urdf.read_bytes())
 tree=ET.parse(r/'research'/f'{name}-collision-before-fine.urdf');link=tree.find('.//link');template=copy.deepcopy(link.find('collision'))
 for node in link.findall('collision'):link.remove(node)
 parts=[]
 def add(vertices,label):
  part=root/f'{name}_hollow_{label}.obj';trimesh.convex.convex_hull(vertices).export(part);node=copy.deepcopy(template);node.find('geometry/mesh').set('filename','/episode/reconstruction/mesh_priors/'+part.name);link.append(node);parts.append(part.name)
 for layer,(lo,hi) in enumerate(zip(knots[:-1],knots[1:])):
  for j in range(segments):
   points=[]
   for h in [lo,hi]:
    for rad in [radius+slope*h-wall,radius+slope*h]:
     for angle in [j*2*np.pi/segments,(j+1)*2*np.pi/segments]:points.append(origin+axis*h+rad*(np.cos(angle)*u+np.sin(angle)*w))
   add(np.array(points),f'{layer:02d}_{j:02d}')
 points=[]
 for h in [bottom,bottom+wall]:
  for angle in np.linspace(0,2*np.pi,segments,endpoint=False):points.append(origin+axis*h+(radius+slope*h)*(np.cos(angle)*u+np.sin(angle)*w))
 add(np.array(points),'base');tree.write(urdf,encoding='unicode')
 report[name]={'collision_pieces':len(parts),'wall_thickness_assumption_m':wall,'fitted_frustum_parameters':fit.x.tolist(),
  'outer_radius_at_axis_origin_m':float(radius),'axial_bounds_m':[float(bottom),float(top)],
  'median_scan_surface_error_m':float(np.median(abs(residual(fit.x)))),
  'p95_scan_surface_error_m':float(np.percentile(abs(residual(fit.x)),95)),
  'visual_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'released_visual_mesh_mass_and_inertia_unchanged':True,
  'geometry_approximation':'48 radial shell segments, four height intervals and a closed 1 mm bottom; cavity remains physically open.'}
 print(name,report[name],flush=True)
(r/'research/hollow-cup-collision-candidate.json').write_text(json.dumps({'method':__doc__,'approved_for_training':False,'diagnostic_only':True,'objects':report},indent=2))
