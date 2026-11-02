"""Validate actual native reference windows without creating physics or training."""
from pathlib import Path
import json
import numpy as np
from flash_chord.data import load_reference,frame_window
ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
reference=ROOT/f"human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave"
full=load_reference(reference,control_fps=20.,motion_speed=.5)
assert full.num_frames==2*(cfg['frames']-1)
source_end=min(cfg['frames']-1,max(g['start'] for g in cfg['grasps'])+22)
end=int(round(source_end*(full.num_frames-1)/(cfg['frames']-1)))+1
prefix=frame_window(full,0,end)
assert prefix.num_frames==end
assert full.object_body_names()==cfg['objects']==prefix.object_body_names()
for method in ['object_body_pos_w','object_body_quat_w']:
    assert np.array_equal(getattr(prefix,method)(),getattr(full,method)()[:end])
for side in ['left','right']:
    for method in ['wrist_pos_w','wrist_quat_w','finger_joint_pos','contact_pos_w','contact_part_ids']:
        assert np.array_equal(getattr(prefix,method)(side),getattr(full,method)(side)[:end])
assert frame_window(full,0,-1) is full
print(json.dumps({'passed':True,'source_frames':cfg['frames'],'full_native_control_frames':full.num_frames,
 'prefix_last_source_frame':source_end,'prefix_native_control_frames':prefix.num_frames,
 'prefix_is_exact_original_contiguous_window':True,'full_reference_unchanged':True,'policy_training_started':False}))
