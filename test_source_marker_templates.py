import json
from pathlib import Path
import tempfile
import unittest

from improvement_common import digest
from source_marker_templates import load


class SourceMarkerTemplateTests(unittest.TestCase):
    def fixture(self,root):
        mesh=root/'scan.glb';mesh.write_bytes(b'original scan')
        image=root/'source.png';image.write_bytes(b'original source image')
        markers=[dict(marker_id=i,center_m=[i,.01,.02],radius_m=.006) for i in range(3)]
        template=root/'templates.json';template.write_text(json.dumps(dict(objects={'rack':dict(mesh=str(mesh),mesh_sha256=digest(mesh),markers=markers)})))
        data=dict(template_sha256=digest(template),source_images=[dict(path=str(image),sha256=digest(image))],
                  objects={'rack':dict(mesh_sha256=digest(mesh),retained_marker_ids=[0,2],reason='Source review excludes a rounded plastic post')})
        review=root/'review.json';review.write_text(json.dumps(data))
        return template,review,image,mesh,markers

    def test_source_review_preserves_metric_centres_and_reindexes_feature_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);template,review,_,_,markers=self.fixture(root);original=template.read_bytes()
            result=load(template,review,root/'manifest.json');selected=result['rack']['markers']
            self.assertEqual([m['source_marker_id'] for m in selected],[0,2])
            self.assertEqual([m['marker_id'] for m in selected],[0,1])
            self.assertEqual(selected[1]['center_m'],markers[2]['center_m'])
            self.assertEqual(template.read_bytes(),original)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['objects']['rack']['rejected_source_marker_ids'],[1])

    def test_missing_review_keeps_original_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);template,_,_,_,markers=self.fixture(root)
            self.assertEqual(load(template,root/'absent.json')['rack']['markers'],markers)

    def test_changed_template_cannot_reuse_reviewed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);template,review,*_=self.fixture(root);template.write_text(template.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'another template'):load(template,review)

    def test_changed_mesh_or_source_image_cannot_reuse_review(self):
        for which in ['mesh','image']:
            with self.subTest(which=which),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);template,review,image,mesh,_=self.fixture(root)
                (mesh if which=='mesh' else image).write_bytes(b'changed input')
                with self.assertRaises(ValueError):load(template,review)

    def test_source_evidence_is_required_before_rejecting_curvature_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);template,review,*_=self.fixture(root);data=json.loads(review.read_text());data['source_images']=[];review.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'source image evidence'):load(template,review)
