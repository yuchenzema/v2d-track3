"""Bound temporary crop-warp memory while preserving all pose hypotheses."""
import copy,runpy,sys
import torch
import kornia.geometry.transform as geometry
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
original=geometry.warp_perspective
LIMIT=21
def bounded_warp(src,M,dsize,*args,**kwargs):
 if len(src)<=LIMIT:return original(src,M,dsize,*args,**kwargs)
 return torch.cat([original(src[start:start+LIMIT],M[start:start+LIMIT] if len(M)==len(src) else M,dsize,*args,**kwargs) for start in range(0,len(src),LIMIT)],dim=0)
if '--self-test' in sys.argv:
 torch.manual_seed(42);src=torch.rand((47,3,48,64),device='cuda');M=torch.eye(3,device='cuda').repeat(47,1,1);M[:,0,2]=torch.linspace(-3,3,47,device='cuda');M[:,1,2]=1.25
 expected=original(src,M,(31,43),align_corners=False);actual=bounded_warp(src,M,(31,43),align_corners=False);error=float((expected-actual).abs().max());assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-6),error
 print({'crop_warp_max_absolute_difference':error,'hypothesis_count_preserved':len(actual),'chunk_limit':LIMIT});sys.exit(0)
geometry.warp_perspective=bounded_warp
import v2d.foundation_pose.lib.trt_predictors as predictors
def join_parts(parts):
 result=copy.copy(parts[0])
 for name,value in vars(parts[0]).items():
  if not isinstance(value,torch.Tensor):continue
  values=[getattr(part,name) for part in parts]
  if all(v.ndim>0 and len(v)==len(part.rgbAs) for v,part in zip(values,parts)):
   setattr(result,name,torch.cat(values,dim=0))
  else:
   assert all(v.shape==value.shape and torch.equal(v,value) for v in values),(name,[v.shape for v in values])
 return result
class BoundedDataset:
 def __init__(self,dataset):self.dataset=dataset
 def transform_batch(self,batch,**kwargs):
  count=len(batch.rgbAs)
  if count<=LIMIT:return self.dataset.transform_batch(batch,**kwargs)
  parts=[]
  for start in range(0,count,LIMIT):
   part=copy.copy(batch)
   for name,value in vars(batch).items():
    if isinstance(value,torch.Tensor) and value.ndim and len(value)==count:setattr(part,name,value[start:start+LIMIT])
   parts.append(self.dataset.transform_batch(part,**kwargs))
  return join_parts(parts)
def bounded_crops(function):
 def call(render_size,ob_in_cams,*args,**kwargs):
  # Keep the full render batch: splitting nvdiffrast changes edge rasterization.
  # Only the unwarped full-resolution depth expansion needs bounded batches.
  if 'dataset' in kwargs:kwargs['dataset']=BoundedDataset(kwargs['dataset'])
  return function(render_size,ob_in_cams,*args,**kwargs)
 return call
if '--crop-self-test' in sys.argv:
 import numpy as np,trimesh,nvdiffrast.torch as dr
 from learning.datasets.h5_dataset import PoseRefinePairH5Dataset,ScoreMultiPairH5Dataset
 from Utils import make_mesh_tensors,depth2xyzmap
 torch.manual_seed(42);rng=np.random.default_rng(42);mesh=trimesh.creation.box(extents=[.1,.1,.1]);cfg=predictors._base_config();rgb=rng.integers(0,255,(48,64,3),dtype=np.uint8);depth=np.ones((48,64),np.float32);K=np.array([[60.,0,32],[0,60.,24],[0,0,1]],np.float32);poses=torch.eye(4,device='cuda').repeat(23,1,1);poses[:,2,3]=1.;poses[:,0,3]=torch.linspace(-.1,.1,23,device='cuda');common=dict(mesh=mesh,rgb=torch.as_tensor(rgb,device='cuda').float(),depth=torch.as_tensor(depth,device='cuda'),K=K,crop_ratio=cfg.crop_ratio,mesh_diameter=.17320508,cfg=cfg,mesh_tensors=make_mesh_tensors(mesh),glctx=dr.RasterizeCudaContext())
 for name in ['make_refine_crop_data','make_score_crop_data']:
  function=getattr(predictors,name);options=dict(common)
  if name=='make_refine_crop_data':options.update(dataset=PoseRefinePairH5Dataset(cfg=cfg,h5_file='',mode='test'),xyz_map=torch.as_tensor(depth2xyzmap(depth,K),device='cuda'))
  else:options['dataset']=ScoreMultiPairH5Dataset(cfg=cfg,mode='test',h5_file=None,max_num_key=1)
  expected=function(cfg.input_resize,poses,**options);actual=bounded_crops(function)(cfg.input_resize,poses,**options);errors={}
  for attr in ['rgbAs','rgbBs','xyz_mapAs','xyz_mapBs','poseA']:
   x,y=getattr(expected,attr),getattr(actual,attr);errors[attr]={'maximum':float((x-y).abs().max()),'mean':float((x-y).abs().mean()),'shape':list(x.shape)}
  for attr in ['tf_to_crops','poseA','rgbAs','rgbBs','xyz_mapAs','xyz_mapBs']:
   assert torch.allclose(getattr(expected,attr),getattr(actual,attr),atol=2e-5,rtol=2e-5),attr
  print('CROP_NUMERICAL_TEST',name,errors,flush=True)
 sys.exit(0)
predictors.make_refine_crop_data=bounded_crops(predictors.make_refine_crop_data)
predictors.make_score_crop_data=bounded_crops(predictors.make_score_crop_data)
print('FoundationPose crop warps bounded to 21 hypotheses; full-resolution inputs and all rotation hypotheses preserved.',flush=True)
# Resume only poses produced by this exact video/depth/mesh stage. Every cached
# pose is loaded and installed in the tracker; uncached frames still run FP.
import importlib.util
from pathlib import Path
spec=importlib.util.find_spec('v2d.foundation_pose.lib.run_video_to_poses')
source=Path(spec.origin).read_text()
# Reject reuse if a subsequent invocation changes any source identity or option.
import hashlib,json
options=dict(zip(sys.argv[1::2],sys.argv[2::2]));pose_folder=Path(options['--poses_dir']);identity_path=pose_folder.with_suffix('.resume-identity.json');identity={'arguments':sys.argv[1:],'inputs':{}}
for flag in ['--video_path','--camera_intrinsics_path','--mesh_path']:
 path=Path(options[flag]);identity['inputs'][flag]=hashlib.sha256(path.read_bytes()).hexdigest()
if identity_path.exists():
 assert json.loads(identity_path.read_text())==identity,'Pose inputs changed: use a new output directory or explicitly archive obsolete poses'
else:
 identity_path.parent.mkdir(parents=True,exist_ok=True);identity_path.write_text(json.dumps(identity,indent=2))

old="initial_pose = tracker.register(rgb, _apply_mask(depth, mask) if mask_depth else depth, mask, scaled_intrinsics, iteration=register_iteration)"
replacement="cached_initial=Path(poses_dir)/f'{reference_frame:06d}.json'\n    initial_pose = Transform3d.load(str(cached_initial)) if cached_initial.exists() else tracker.register(rgb, _apply_mask(depth, mask) if mask_depth else depth, mask, scaled_intrinsics, iteration=register_iteration)"
assert old in source;source=source.replace(old,replacement)
for direction,indent in [('Forward','        '),('Backward','            ')]:
 old=indent+f'logger.info(f"{direction} frame {{frame_idx}}/{{num_frames}}")'
 new=indent+"cached=Path(poses_dir)/f'{frame_idx:06d}.json'\n"+indent+"if cached.exists():\n"+indent+"    pose=Transform3d.load(str(cached));tracker.reset_to_pose(pose);continue\n"+old
 assert old in source;source=source.replace(old,new)
exec(compile(source,spec.origin,'exec'),{'__name__':'__main__','__file__':spec.origin})
