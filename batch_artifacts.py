"""Recognize completed deliverables from the validated artifact set."""
import hashlib,json
from pathlib import Path
from episode_clock import control_frames

VIDEOS=('original_vs_policy.mp4','original_vs_reconstruction.mp4','policy_rollout_source_timed.mp4','trained_policy_source_camera.mp4','trained_policy_rollout.mp4','reconstruction_reference.mp4')
REQUIRED=('original_vs_policy.mp4','original_vs_reconstruction.mp4','rollout_source_timed.mp4','rollout_3d.mp4','scene.glb','policy.safetensors','training_state.safetensors','policy_states.npz','report.json')

def problems(root,verify_hashes=False,require_source_review=True):
 r=Path(root)
 try:
  c=json.loads((r/'episode-config.json').read_text());report=json.loads((r/'result-report.json').read_text());manifest=json.loads((r/'deliverables/manifest.json').read_text());issues=[]
  if not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():issues.append('training completion marker missing')
  checks={v['video']:v for v in report['validation']}
  for index,name in enumerate(VIDEOS):
   path=r/'videos'/name;expected=c['frames'] if index<3 else control_frames(c);v=checks.get(name,{})
   if not path.is_file() or path.stat().st_size==0 or v.get('frames')!=expected or abs(v.get('fps',0)-20)>.01:issues.append('video missing or not frame-validated: '+name)
  for name in REQUIRED:
   if name not in manifest:issues.append('manifest entry missing: '+name)
  for name,entry in manifest.items():
   path=r/entry['path']
   if not path.is_file() or path.stat().st_size!=entry['bytes']:issues.append('artifact missing or size mismatch: '+name)
   elif verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:issues.append('artifact hash mismatch: '+name)
  if report['protocol']['object_assistance']!=0 or report['protocol']['failure_resets']:issues.append('unassisted continuous attempt protocol violated')
  if require_source_review and not report['all_original_frames_visually_reviewed']:issues.append('source review incomplete')
  return issues
 except (OSError,ValueError,KeyError,TypeError) as exc:return [str(exc)]

def complete(root):return not problems(root)
