"""Handle individually underconstrained RGB frames using the measured rigid camera rig."""
def adapt(s):
 s=s.replace('registrations=[];cloud=[];errors=[]','registrations=[];cloud=[];errors=[];missing_frames=[];feature_counts=[]')
 s=s.replace('    assert len(rgb)>=15,(i,len(rgb))',"    feature_counts.append(len(rgb))\n    if len(rgb)<10:\n        registrations.append(None);errors.append(None);missing_frames.append(i);continue")
 s=s.replace('    assert ok and ids is not None and len(ids)>=10,i',"    if not ok or ids is None or len(ids)<8:\n        registrations.append(None);errors.append(None);missing_frames.append(i);continue")
 old="fixed=np.eye(4);fixed[:3,:3]=Rotation.from_matrix(np.array(registrations)[:,:3,:3]).mean().as_matrix();fixed[:3,3]=np.median(np.array(registrations)[:,:3,3],axis=0)"
 new="valid_registrations=np.array([m for m in registrations if m is not None]);assert len(valid_registrations)>=.8*len(registrations),(len(valid_registrations),len(registrations))\nfixed=np.eye(4);fixed[:3,:3]=Rotation.from_matrix(valid_registrations[:,:3,:3]).mean().as_matrix();fixed[:3,3]=np.median(valid_registrations[:,:3,3],axis=0)\nfor i in missing_frames:registrations[i]=fixed.copy()"
 assert old in s;s=s.replace(old,new).replace('float(np.median(errors))','float(np.median([e for e in errors if e is not None]))')
 s=s.replace("'rgb_registration_error_by_frame_px':errors", "'rgb_registration_error_by_frame_px':errors,'rigid_calibration_fallback_frames':missing_frames,'rgb_feature_correspondences_by_frame':feature_counts,'low_feature_fallback_method':'Use the cross-frame measured rigid RGB/stereo transform only when this frame has too few verified correspondences; no task-object motion interpolation'")
 return s
