import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from source_mask_rasterizer import MeshSilhouette
from t3_local_score import compute,load_public
from episode_clock import control_frames,motion_speed
from source_cad_alignment import object_visibility
from training_continuation import extend
from improvement_common import digest
import json
import pyarrow as pa
import pyarrow.parquet as pq
from official_policy_suite import bind_rendered_attempt,POSE_COLUMNS


class SourceClockTests(unittest.TestCase):
    def test_new_policy_preserves_every_source_frame(self):
        cfg=dict(frames=263,validated_video_pipeline=True)
        self.assertEqual(control_frames(cfg),263)
        self.assertEqual(motion_speed(cfg),1.)

    def test_old_recordings_retain_their_declared_clock(self):
        cfg=dict(frames=263)
        self.assertEqual(control_frames(cfg),524)
        self.assertEqual(motion_speed(cfg),.5)


class ContinuationTests(unittest.TestCase):
    def fixture(self,root):
        (root/'research').mkdir();folder=root/'flash_chord_coherent_actor_transfer';folder.mkdir()
        (root/'episode-config.json').write_text(json.dumps(dict(validated_video_pipeline=True,training_curriculum='fresh_rebuild',training_budget=12800000)))
        (root/'reference').write_bytes(b'frozen geometry and reference')
        (root/'coherent-input-checksums.json').write_text(json.dumps({'reference':digest(root/'reference')}))
        for step in [10000000,12800000]:
            for kind in ['policy','state']:(folder/f'{kind}_{step}.safetensors').write_bytes(b'fixture checkpoint pair')
        (folder/'TRAINING_COMPLETED').write_text('previous round complete')

    def test_extension_uses_latest_resumable_step_and_keeps_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            target=extend(root)
            self.assertEqual(target,25600000)
            self.assertEqual((root/'reference').read_bytes(),b'frozen geometry and reference')
            self.assertFalse((root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists())

    def test_changed_geometry_is_rejected_before_extending(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);(root/'reference').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'Frozen reference changed'):extend(root)
            self.assertTrue((root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists())


class OfficialRenderBindingTests(unittest.TestCase):
    def fixture(self,root,different=False):
        (root/'evaluation').mkdir();checkpoint=root/'policy.safetensors';checkpoint.write_bytes(b'test fixture actor')
        sha=digest(checkpoint);official=root/f'evaluation/official/{sha[:16]}';official.mkdir(parents=True)
        poses=np.zeros((2,1,7));poses[:,:,6]=1.
        np.savez(root/'evaluation/record.npz',body_poses=poses,body_names=['track3_white_pot/object'],object_assistance=[0.,0.])
        selected=dict(checkpoint='/episode/policy.safetensors',recorded_states='evaluation/record.npz')
        (root/'final-controller-selection.json').write_text(json.dumps({'selected':selected}))
        data={name:poses[:,0,index].copy() for index,name in enumerate(POSE_COLUMNS)}
        if different:data['achieved_position_x'][1]=.01
        data.update(reference_frame=[0,1],object_body_name=['white_pot','white_pot'])
        table=pa.table(data).replace_schema_metadata({b'flash_chord.step_count':b'2'})
        parquet=official/'recorded.parquet';pq.write_table(table,parquet)
        (official/'suite.json').write_text(json.dumps({'recorded_trajectory':str(parquet)}))

    def test_public_object_names_bind_to_actual_simulator_bodies(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            self.assertTrue(bind_rendered_attempt(root)['rendered_objects_match_official_recording'])

    def test_different_rendered_motion_blocks_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root,different=True)
            with self.assertRaisesRegex(AssertionError,'publication blocked'):bind_rendered_attempt(root)


class DepthRenderingTests(unittest.TestCase):
    def test_overlapping_lid_stays_visible_above_the_pot(self):
        masks=[np.ones((4,4),bool),np.ones((4,4),bool)]
        depths=[np.full((4,4),.7),np.full((4,4),.65)]
        self.assertTrue(object_visibility(masks,depths,1).all())
        self.assertFalse(object_visibility(masks,depths,0).any())

    def test_front_surface_wins_independent_of_triangle_order(self):
        triangle=np.array([[-.2,-.2,1.],[.2,-.2,1.],[0.,.2,1.]])
        points=np.r_[triangle,triangle*2]
        intr=dict(fx=600,fy=600,cx=512,cy=384)
        for faces in [np.array([[0,1,2],[3,4,5]]),np.array([[3,4,5],[0,1,2]])]:
            renderer=MeshSilhouette(points,faces,intr);depth=renderer.depth(np.eye(4))
            self.assertAlmostEqual(float(depth[192,256]),1.,places=6)
            np.testing.assert_array_equal(depth>0,renderer.render(np.eye(4)))
            self.assertEqual(depth[0,0],0.)

    def test_depth_is_metric_and_behind_camera_is_excluded(self):
        points=np.array([[-.2,-.2,1.],[.2,-.2,1.],[0.,.2,1.]])
        renderer=MeshSilhouette(points,[[0,1,2]],dict(fx=600,fy=600,cx=512,cy=384))
        pose=np.eye(4);pose[2,3]=.4
        self.assertAlmostEqual(float(renderer.depth(pose)[192,256]),1.4,places=6)
        pose[2,3]=-2
        self.assertFalse(renderer.depth(pose).any())


class OfficialPublicMetricControls(unittest.TestCase):
    def prediction(self,transform=None,error=False):
        names,poses,valid,_=load_public(0)
        poses=poses.copy();poses[~valid,3:]=[0,0,0,1]
        if transform is not None:
            poses[:,:,:3]=poses[:,:,:3]@transform[:3,:3].T+transform[:3,3]
            poses[:,:,3:]=(Rotation.from_matrix(transform[:3,:3])*Rotation.from_quat(poses[:,:,3:].reshape(-1,4))).as_quat().reshape(poses[:,:,3:].shape)
        if error:poses[1:,0,0]+=.20
        return poses

    def score(self,poses):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'prediction.npz';np.savez(file,poses_xyzw=poses,object_names=np.array(load_public(0)[0]))
            return compute(0,prediction=file,sample_count=100,include_mesh=False)

    def test_scene_alignment_is_invariant_to_global_rigid_frame(self):
        transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[.3,-.2,.8]).as_matrix();transform[:3,3]=[.5,-.4,1.2]
        report=self.score(self.prediction(transform))
        self.assertAlmostEqual(report['metrics']['AUC'],1.,places=5)
        self.assertLess(report['metrics']['MPPE_cm'],1e-4)
        self.assertEqual(report['kind'],'video_reconstructed_reference_not_policy')
        self.assertFalse(report['exact_kaggle_score'])

    def test_later_trajectory_error_is_not_removed_by_frame_zero_alignment(self):
        report=self.score(self.prediction(error=True))
        self.assertLess(report['metrics']['AUC'],.7)
        self.assertEqual(report['metrics']['MP-SR'],0.)
        self.assertGreater(report['metrics']['RPE_cm'],10.)


if __name__=='__main__':unittest.main()
