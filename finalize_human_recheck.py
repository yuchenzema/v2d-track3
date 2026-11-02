"""Add visual review findings and motion/contact diagnostics to the completed all20 audit."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc
from thorough_episode_recheck import project,matrix,artifact
OUT=BATCH/'human_video_recheck_20261021'
# These findings were inspected directly in the generated original/policy timelines and worst-frame sheets.
NOTES={
1:('medium',[116,131,256,328],'Pot and lid complete the broad task, but pot lift height and lid placement differ. Some low-overlap lid frames are heavily occluded and cannot establish a visible failure.','Refine pot lift trajectory, lid contact placement and hand withdrawal; preserve the user-preferred policy.'),
3:('high',[99,116,125,177],'The spoon does not follow the human stirring path inside the pot; spoon height and orientation differ while the pot remains grasped.','Recover the repeated spoon-head motion and pot-relative contact, then train for those motions rather than only the broad object pose.'),
5:('high',[61,112,150,174],'The broad pot lift is present, but repeated stirring is much flatter and more stationary than the human motion; the spoon head is near the rim in mismatched frames.','Refine time-resolved spoon-head trajectory and grasp orientation, including stirring while the pot is lifted.'),
6:('high',[77,120,132,156],'The spoon remains near a fixed rim pose through much of the source stirring motion. Tool orientation and right-hand pose diverge.','Reconstruct the spoon-head path and stabilize its grasp during repeated inside-pot motion.'),
8:('high',[71,105,119,210,218],'Pouring gestures broadly occur, but pot/cup alignment drifts and the cup is displaced during final hand withdrawal instead of staying upright in the human resting position.','Correct cup placement and release sequencing; check relative cup-to-pot pose during both pours.'),
10:('medium',[78,89,100,130],'Cup stacking broadly matches and remains upright; cup seating and hand contact/release timing still differ.','Refine nesting depth, cup alignment and the hand release sequence.'),
14:('medium',[78,89,100,110],'Cup transfer and inverted placement broadly occur, but cup alignment and hand contact after the transfer do not match precisely.','Improve transfer contact geometry, cup angle in the rack and post-placement hand timing.'),
15:('medium',[67,80,94,114],'The spoon reaches the utensil holder, but its tilt and hand support differ during insertion; the empty hand also follows a different path.','Refine insertion angle, transient support and withdrawal timing while retaining the stable final placement.'),
16:('medium',[33,67,77,89],'The handover/insertion sequence is present. Spoon height and tilt differ around the transfer, and the supporting hand moves differently.','Improve handover synchronization and spoon pose through the transition into the holder.'),
19:('critical',[50,59,69,78,106,213],'The brush loses correspondence with the human sweep, separates from the acting hand and rests on the table while the hand continues. The dustpan also drifts.','Rebuild and validate brush grasp/contact before full-motion retraining; preserve the dustpan brace throughout sweeping.'),
20:('high',[102,175,184,193],'Pitcher motion is broadly present, but the cup tips and moves during final release while the human cup stays upright.','Correct cup support and release first; then refine pitcher/cup alignment during pouring.'),
25:('critical',[123,161,315,339,348],'The two wooden pieces fail to preserve the human interlocked cross assembly. They separate and tip during the final release.','Validate slot collision geometry and insertion feasibility, then train successful insertion and stable assembled release.'),
28:('critical',[80,89,98,107,179,219],'The pot separates from the hand during the lift/transfer and falls onto or away from the rack. The final human placement is not reproduced.','Repair handle grasp and transfer contact; validate uninterrupted lifting before learning inverted placement and release.'),
30:('critical',[57,66,75,84,191,262],'The published brush leaves the human pickup/working pose and remains misoriented or unsupported relative to the rack; later brushing and final placement differ.','Use the corrected pickup reconstruction, establish a stable continuous brush grasp and rack brace, then restore source brushing and release timing.'),
32:('high',[93,154,211,258],'The accepted controller is a relative improvement, but spoon/cup contact still differs during stirring. The cup is displaced/tipped at withdrawal and the spoon final pose differs.','Refine spoon-head motion inside the cup and stabilize both final placements before hand withdrawal.'),
33:('critical',[59,68,101,110,158,347],'The spoon is left on the table while the human lifts and stirs with it; the robot hand continues without the tool. Cup release also differs.','Repair spoon pickup and sustained grip before training stirring; verify uninterrupted contact in a full attempt.'),
35:('medium',[95,115,142,275],'Repeated handovers broadly occur, but spoon orientation and hand-to-hand timing differ, including upright and horizontal transitions.','Refine handover timing and tool axial orientation without losing the existing repeated transfer sequence.'),
36:('critical',[163,219,258,297,306,332],'The spoon is dropped or rests on the table during later handovers while the human still holds it. Its final orientation is wrong.','Reconstruct hand identity and handover contacts, then train uninterrupted alternating grasps through the late sequence.'),
37:('medium',[6,35,64,78],'The lift-and-return task broadly matches, but the spoon height and angle differ and there is a small initial resting alignment error.','Refine initial support, lift height and final placement; retain the stable grasp.'),
38:('lower',[44,153,166,253],'The dustpan follows the broad source motion and has no frames below the local 50% overlap threshold. Small grip and pose differences remain; this is not a certificate of exact human imitation.','Preserve this policy and review small grip/release differences after the failed tasks are repaired.')}

def main():
 summary=json.loads((OUT/'summary.json').read_text());assert summary['episodes_checked']==20 and summary['source_frames_checked']==5181;rows=[]
 for old in summary['episodes']:
  ep=old['episode'];dst=OUT/f'episode_{ep:06d}';r=json.loads((dst/'report.json').read_text());payload=json.loads((dst/'frame-metrics.json').read_text());frames=payload['frames'];root=Path(r['reconstruction_used']);rec=root/'reconstruction';severity,evidence,finding,focus=NOTES[ep]
  r['visual_review']={'reviewed_utc':utc(),'severity':severity,'finding':finding,'correction_focus':focus,'evidence_source_frames':evidence,'timeline_samples_inspected':12,'additional_worst_frames_inspected':4,'every_frame_scored_automatically':True,'all_frames_manually_inspected':False};r['links']['timeline_overview']=artifact(dst/'timeline-overview.jpg')
  for name,obj in r['objects'].items():
   vals=[f['objects'][name] for f in frames];scores=[v['rollout']['iou'] for v in vals];obj.update(initial_1s_mean_iou=float(np.mean(scores[:20])),final_1s_mean_iou=float(np.mean(scores[-20:])),reconstruction_frames_below_0_5_iou=sum(v['reconstruction']['iou']<.5 for v in vals),clear_policy_drift_frame_count=sum(v['reconstruction']['iou']>=.6 and v['rollout']['iou']<.5 and v['rollout']['visible_source_pixels']>=30 for v in vals),reconstruction_or_calibration_review_frame_count=sum(v['reconstruction']['iou']<.5 and v['reconstruction']['visible_source_pixels']>=30 for v in vals))
  visible_bad={i for f in frames for i in [f['frame']] if any(v['rollout']['iou']<.5 and v['rollout']['visible_source_pixels']>=30 for v in f['objects'].values())};r['visible_frames_with_any_object_below_0_5_iou']=len(visible_bad)
  names=list(r['objects'])
  if len(names)==2:
   errors=[]
   for f in frames:
    a,b=[f['objects'][n] for n in names];actual=np.array(a['rollout_position_world_m'])-b['rollout_position_world_m'];expected=np.array(a['reconstruction_position_world_m'])-b['reconstruction_position_world_m'];error=float(np.linalg.norm(actual-expected));f['pair_relative_position_error_m']=error;errors.append(error)
   r['two_object_relative_motion']={'objects':names,'mean_position_error_m':float(np.mean(errors)),'max_position_error_m':float(max(errors)),'meaning':'Difference in the relative displacement between the two physical objects and the video reconstruction at the same source frame; no time alignment optimization.'}
  landmark=rec/'spoon-geometric-tracking.json'
  if landmark.exists() and 'wooden_spoon' in names:
   data=json.loads(landmark.read_text());points=np.array(data['canonical_head_handle_landmarks_m']);k=json.loads((rec/'camera_intrinsics.json').read_text());motion=np.load(rec/'scene_motion_physical.npz');stateinfo=json.loads((OUT/'gui-release-snapshot.json').read_text());release=next(x for x in stateinfo['episodes'] if x['episode']==ep)['videos']['latest'];state=np.load(release['recorded_states']);index=state['body_names'].tolist().index('track3_wooden_spoon/object');errors=[]
   for f in frames:
    i=f['frame'];M=np.linalg.inv(motion['camera_to_world'][i])@matrix(state['body_poses'][f['recorded_control_frame'],index]);uv=project(points@M[:3,:3].T+M[:3,3],k);measurement=data['frames'][i];error=float(np.linalg.norm(uv[0]-measurement['head_uv']));tail=float(np.linalg.norm(uv[1]-measurement['handle_end_uv']));f['spoon_landmarks']={'head_error_px':error,'handle_end_error_px':tail};errors.append(error)
   r['spoon_head_source_landmark_check']={'all_source_frames_checked':len(errors),'mean_error_px':float(np.mean(errors)),'max_error_px':float(max(errors)),'frames_above_20px':sum(e>20 for e in errors),'method':'Recorded spoon-head projection compared directly with source-video geometric head landmarks; source tracking uncertainty remains.'}
  save(dst/'frame-metrics.json',payload);r['updated_utc']=utc();save(dst/'report.json',r);rows.append(r)
 summary['episodes']=rows;summary.update(updated_utc=utc(),visual_timeline_and_failure_samples_reviewed_for_all_20=True,total_automatically_scored_source_frames=5181,source_frames_manually_inspected_claimed=False,correction_priority=[{'episode':r['episode'],'severity':r['visual_review']['severity'],'finding':r['visual_review']['finding'],'correction_focus':r['visual_review']['correction_focus']} for severity in ['critical','high','medium','lower'] for r in rows if r['visual_review']['severity']==severity]);save(OUT/'summary.json',summary)
 status=json.loads((OUT/'status.json').read_text());status.update(updated_utc=utc(),episodes=rows,visual_review_completed=True,summary_report=artifact(OUT/'summary.json'));save(OUT/'status.json',status)
 save(OUT/'correction-priorities.json',{'updated_utc':utc(),'official_challenge_scores':False,'no_new_policy_training_performed':True,'priorities':summary['correction_priority']});print('Visual findings, per-frame source landmarks, phase attribution and correction priorities saved for all20.')
if __name__=='__main__':main()
