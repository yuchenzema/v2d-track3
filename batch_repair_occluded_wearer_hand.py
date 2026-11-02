"""Back up and repair only a visually confirmed bystander identity error."""
import fcntl,json,shutil,time
from batch_worker import BASE,context
from batch_native import containers,docker
r,c=context(16)
with (r/'research/collision-reference.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists();assert json.loads((r/'research/terminal-support-reference-selection.json').read_text())['accepted'];backup=r/f'research/occluded-wearer-hand-{time.time_ns()}';backup.mkdir()
 for name in ['scene_motion_physical.npz','hand_measurements.npz']:shutil.copy2(r/'reconstruction'/name,backup/name)
 for reference in list(dict.fromkeys([c['training_reference'],'track3_video','track3_coherent_variant','track3_terminal_supported_candidate'])):
  path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));shutil.copy2(path,backup/f'{reference}.parquet')
 shutil.copy2(BASE/'repair_occluded_wearer_hand.py',r/'scripts/repair_occluded_wearer_hand.py');newton,retarget=containers(r,16)
 try:docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/repair_occluded_wearer_hand.py'],r/'logs/occluded-wearer-hand-repair.log')
 except BaseException:
  for name in ['scene_motion_physical.npz','hand_measurements.npz']:
   path=r/'reconstruction'/name;tmp=path.with_suffix('.restore.tmp');shutil.copy2(backup/name,tmp);tmp.replace(path)
  for reference in list(dict.fromkeys([c['training_reference'],'track3_video','track3_coherent_variant','track3_terminal_supported_candidate'])):
   path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));tmp=path.with_suffix('.restore.tmp');shutil.copy2(backup/f'{reference}.parquet',tmp);tmp.replace(path)
  raise
 shutil.copy2(BASE/'native_reference_audit.py',r/'scripts/native_reference_audit.py');docker(['exec',newton,'env',f"V2D_AUDIT_REFERENCE={c['training_reference']}",'python','-u','/episode/scripts/native_reference_audit.py'],r/'logs/occluded-wearer-hand-collision-audit.log');print('Wearer left-hand occlusion repaired',backup,flush=True)
