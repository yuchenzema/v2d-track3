import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from improvement_common import digest
import experimental_training_admission as admission
from episode_completion_gate import completion


class ExperimentalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'reconstruction').mkdir()
        self.source=self.root/'source.mp4';self.source.write_bytes(b'fixed source')
        self.trajectory=self.root/'reconstruction/cad-source-trajectory.npz';self.trajectory.write_bytes(b'fixed poses')
        (self.root/'episode-config.json').write_text(json.dumps(dict(source=str(self.source),frames=20,objects=['cup'])))
        self.rows=[dict(frame=f,object='cup',visible_pixels=100,iou=.87,centroid_error_px=4.,boundary_error_px=4.,sphere_error_px=2.) for f in range(20)]
        for row in self.rows[-2:]:row['iou']=.8
        self.frames=self.root/'frames.json';self.audit=self.root/'audit.json'
        self.report=dict(episode=38,run=str(self.root),all_source_frames_checked=True,
            unobservable_object_frames=0,final_scene_occlusion_rechecked=True,source_time_warping_used=False,
            physical_projection_applied=False,trajectory_sha256=digest(self.trajectory),frames_report=str(self.frames),
            source_video_sha256=digest(self.source),object_frames_checked=20,final_scene_audit=str(self.audit))
        self.policy=dict(enabled=True,episodes=[38],minimum_iou=.85,maximum_centroid_error_px=5.,
                         maximum_boundary_error_px=5.,minimum_qualified_fraction=.85)
        self.bind()

    def bind(self):
        self.frames.write_text(json.dumps(self.rows));self.report['frames_report_sha256']=digest(self.frames)
        self.audit.write_text(json.dumps(dict(all_object_frames_rechecked=True,
            trajectory_sha256=self.report['trajectory_sha256'],frames_report_sha256=self.report['frames_report_sha256'])))

    def test_relaxed_image_admission_is_explicitly_experimental(self):
        result=admission.assess(self.report,self.policy)
        self.assertTrue(result['admitted']);self.assertFalse(result['strict_completion_granted'])
        self.assertEqual(result['minimum_qualified_fraction'],.9)
        self.assertIn('Experimental training',completion(dict(experimental_only=True))['blocking_reasons'][0])

    def test_hidden_frames_and_changed_source_still_block(self):
        self.report['unobservable_object_frames']=1
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])
        self.report['unobservable_object_frames']=0;self.source.write_bytes(b'changed')
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])

    def test_each_object_needs_qualifying_fraction_not_aggregate_mean(self):
        (self.root/'episode-config.json').write_text(json.dumps(dict(source=str(self.source),frames=20,objects=['cup','lid'])))
        self.rows += [dict(row,object='lid',iou=.7) for row in self.rows]
        self.report['object_frames_checked']=40;self.bind()
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])

    def test_duplicate_missing_frame_and_stale_hash_are_rejected(self):
        self.rows[-1]=dict(self.rows[0]);self.bind()
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])
        self.trajectory.write_bytes(b'changed poses')
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])

    def test_sphere_threshold_stays_three_pixels_and_bad_evidence_does_not_disappear(self):
        for row in self.rows[:4]:row['sphere_error_px']=4.
        self.bind();self.assertFalse(admission.assess(self.report,self.policy)['admitted'])
        self.rows[0]['centroid_error_px']=None;self.bind()
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])

    def test_disabled_policy_and_out_of_scope_episode_are_rejected(self):
        for policy in [dict(self.policy,enabled=False),dict(self.policy,episodes=[1])]:
            self.assertFalse(admission.assess(self.report,policy)['admitted'])

    def test_physical_projection_needs_explicit_experimental_scope_and_fresh_source_evidence(self):
        self.report.update(physical_projection_applied=True)
        self.assertFalse(admission.assess(self.report,self.policy)['admitted'])
        policy=dict(self.policy,allow_source_revalidated_physical_candidate=True)
        self.assertFalse(admission.assess(self.report,policy)['admitted'])
        self.report.update(reference_kind='source_revalidated_physical_candidate',experimental_only=True)
        self.assertTrue(admission.assess(self.report,policy)['admitted'])
        self.frames.write_text('[]')
        self.assertFalse(admission.assess(self.report,policy)['admitted'])

    def test_camera_ray_clearance_preserves_origin_projection_and_requested_height(self):
        import numpy as np
        from source_ray_clearance_candidate import lift_along_source_rays
        objects=np.tile(np.eye(4),(3,1,1,1));cameras=np.tile(np.eye(4),(3,1,1))
        cameras[:,2,3]=1.;objects[:,0,:3,3]=[.1,.2,.5];before=objects.copy()
        lift=np.array([[.01],[0.],[.015]])
        moved=lift_along_source_rays(objects,cameras,lift)
        old_ray=objects[:,:,:3,3]-cameras[:,None,:3,3]
        new_ray=moved[:,:,:3,3]-cameras[:,None,:3,3]
        np.testing.assert_allclose(old_ray[:,:,:2]/old_ray[:,:,2:],new_ray[:,:,:2]/new_ray[:,:,2:])
        np.testing.assert_allclose(moved[:,:,2,3]-objects[:,:,2,3],lift)
        np.testing.assert_array_equal(objects,before)
        np.testing.assert_array_equal(moved[:,:,:3,:3],objects[:,:,:3,:3])

    def test_source_metadata_missing_from_cad_is_restored_without_replacing_poses(self):
        from install_source_cad_reference import install_source_measurements
        target=self.root/'target';(target/'reconstruction').mkdir(parents=True)
        source=self.root/'frontend';(source/'reconstruction').mkdir(parents=True)
        (source/'reconstruction/scene-reconstruction.json').write_text('{"table_height_gauge_m":0.65}')
        approved=self.root/'approved';(approved/'reconstruction').mkdir(parents=True)
        poses=target/'reconstruction/scene_motion_physical.npz';poses.write_bytes(b'approved frozen poses')
        self.assertIn('scene-reconstruction.json',install_source_measurements(target,approved,source))
        self.assertEqual(poses.read_bytes(),b'approved frozen poses')
        self.assertEqual(install_source_measurements(target,approved,source),[])

    def test_host_symlinks_are_materialized_for_container_access_without_mutating_sources(self):
        from install_source_cad_reference import install_source_measurements
        target=self.root/'target';(target/'reconstruction').mkdir(parents=True)
        source=self.root/'frontend';(source/'reconstruction/frames').mkdir(parents=True)
        image=source/'reconstruction/frames/000000.png';image.write_bytes(b'original image')
        metadata=source/'reconstruction/scene-reconstruction.json';metadata.write_text('{}')
        for name in ['frames','scene-reconstruction.json']:
            (target/'reconstruction'/name).symlink_to(source/'reconstruction'/name)
        install_source_measurements(target,source,source)
        for name in ['frames','scene-reconstruction.json']:
            self.assertFalse((target/'reconstruction'/name).is_symlink())
        (target/'reconstruction/frames/000000.png').write_bytes(b'local changed')
        self.assertEqual(image.read_bytes(),b'original image')

    def test_unchanged_failed_pilot_is_not_retried_but_repaired_execution_is(self):
        import evaluation_experiment_queue as queue
        out=self.root/'queue';out.mkdir()
        (out/'source-cad-status.json').write_text(json.dumps(dict(results={'38':self.report})))
        (out/'experimental-training-status.json').write_text(json.dumps(dict(attempts={'38':dict(
            trajectory_sha256=self.report['trajectory_sha256'],execution_input_hashes={'installer':'old'})})))
        with patch.object(queue,'execution_inputs',return_value={'installer':'old'}):
            self.assertEqual(queue.candidates(out,[38],self.policy),[])
        with patch.object(queue,'execution_inputs',return_value={'installer':'fixed'}):
            self.assertEqual(queue.candidates(out,[38],self.policy)[0][0],38)


if __name__=='__main__':unittest.main()
