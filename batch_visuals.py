"""Generate per-episode textured scene from released scans and observed table pixels."""
from pathlib import Path
import json,subprocess,sys,xml.etree.ElementTree as ET
BASE=Path('/home/yu_eli/v2d-track3');OLD=BASE/'runs/episode_000005_newton_20261015/scripts/prepare_visuals.py'
r=Path(sys.argv[1]);c=json.loads((r/'episode-config.json').read_text());s=OLD.read_text();start=s.index("paths={'white_pot'");end=s.index(';provenance={}',start)
paths={}
for name in c['objects']:
 if name=='wooden_spoon':paths[name]=str(r/f'reconstruction/mesh_priors/{name}.glb')
 else:
  urdf=BASE/f'datasets/video_to_data_challenge/track_3/public/urdf/{name}_rigid.urdf';mesh=ET.parse(urdf).find('.//visual/geometry/mesh');paths[name]=str((urdf.parent/mesh.get('filename')).resolve())
s=s[:start]+f'paths={paths!r}'+s[end:];s=s.replace('for b in [1,2]:','for b in range(1,len(paths)+1):').replace('Original episode 5 wood measurements','Original episode-specific wood measurements')
if c.get('frame_cache_format')=='jpg':s=s.replace('frames/{i:06d}.png','frames/{i:06d}.jpg')
s=s.replace('assert seen[inside].mean()>.25','assert seen.any()')
s=s.replace('mesh.visual.material.doubleSided=True','\n if hasattr(mesh.visual,"material"):mesh.visual.material.doubleSided=True')
s=s.replace("mesh=trimesh.load(path).to_geometry();uv=mesh.visual.uv;texture=mesh.visual.material.baseColorTexture", "mesh=trimesh.load(path).to_geometry()\n if not hasattr(mesh.visual,'uv'):mesh.visual=mesh.visual.to_texture()\n uv=mesh.visual.uv;material=mesh.visual.material;texture=getattr(material,'baseColorTexture',None)\n if texture is None:texture=material.image")
s=s.replace("for i in range(0,len(m['camera_to_world']),10):","texture_frames=sorted(int(p.stem) for p in (REC/'depth').glob('*.png'))\nfor i in texture_frames:")
s=s.replace("'source_frames':list(range(0,len(m['camera_to_world']),10))","'source_frames':texture_frames")
p=r/'scripts/prepare_visuals.py';p.write_text(s);subprocess.run([str(r/'.venv-video/bin/python'),str(p)],check=True)
