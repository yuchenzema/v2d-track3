"""Explicit soft contact material at brush bristle tips; rigid handle is retained.

This approximates bristle bending through native Newton/MuJoCo contact material.
It does not apply object guidance, change mass/inertia, write object poses or
change robot actuator limits. Only hulls entirely below the declared tip boundary
are eligible; every selected material is recorded for reproducibility.
"""
import json,os
from pathlib import Path
import numpy as np
from newton.solvers import SolverMuJoCo

def install():
 profile=json.loads(os.environ['V2D_BRISTLE_COMPLIANCE']);assert set(profile)=={'stiffness','damping','tip_max_z'}
 assert 10<=profile['stiffness']<=5000 and .1<=profile['damping']<=100 and -.04<=profile['tip_max_z']<=-.01
 assert not getattr(SolverMuJoCo,'_brush_tip_material_installed',False)
 original=SolverMuJoCo.__init__
 def initialize(self,model,*args,**kwargs):
  original_ke=model.shape_material_ke.numpy();ke=original_ke.copy();original_kd=model.shape_material_kd.numpy();kd=original_kd.copy()
  priority_array=model.mujoco.geom_priority;priority=priority_array.numpy().copy();bodies=model.shape_body.numpy();transforms=model.shape_transform.numpy();scales=model.shape_scale.numpy();changed=[]
  from scipy.spatial.transform import Rotation
  for i,body in enumerate(bodies):
   if body<0 or 'track3_mini_sweeper' not in model.body_label[int(body)]:continue
   mesh=model.shape_source[i]
   if mesh is None:continue
   points=np.asarray(mesh.vertices)*scales[i];pose=transforms[i];points=Rotation.from_quat(pose[3:]).apply(points)+pose[:3]
   if float(points[:,2].max())>profile['tip_max_z']:continue
   ke[i]=profile['stiffness'];kd[i]=profile['damping'];priority[i]=2;changed.append(i)
  assert changed,'No convex hulls lie entirely within the declared bristle-tip region'
  untouched=np.ones(len(ke),bool);untouched[changed]=False;np.testing.assert_array_equal(ke[untouched],original_ke[untouched]);np.testing.assert_array_equal(kd[untouched],original_kd[untouched])
  model.shape_material_ke.assign(ke);model.shape_material_kd.assign(kd);priority_array.assign(priority)
  output=os.environ.get('V2D_BRISTLE_MATERIAL_REPORT','/episode/research/brush-tip-material-runtime.json');Path(output).write_text(json.dumps({'profile':profile,'soft_tip_shapes':len(changed),'shape_count':len(ke),'handle_and_other_shape_materials_unchanged':True,'object_mass_inertia_and_actuator_limits_unchanged':True,'native_contact_material_priority':2,'implementation':'Native shape_material_ke/kd and MuJoCo geom_priority; no object control forces.'},indent=2))
  print('Native compliant brush-tip contacts',len(changed),'profile',profile,flush=True)
  original(self,model,*args,**kwargs)
 SolverMuJoCo.__init__=initialize;SolverMuJoCo._brush_tip_material_installed=True
