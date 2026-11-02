import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from batch_artifacts import problems,VIDEOS,REQUIRED

class DiagnosticReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'videos').mkdir();(self.root/'deliverables').mkdir();(self.root/'flash_chord_coherent_actor_transfer').mkdir()
        (self.root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').touch()
        (self.root/'episode-config.json').write_text(json.dumps({'frames':202}))
        for name in VIDEOS:(self.root/'videos'/name).write_bytes(b'video')
        manifest={}
        for name in REQUIRED:
            p=self.root/'deliverables'/name;p.write_bytes(b'artifact')
            manifest[name]={'path':str(p.relative_to(self.root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        (self.root/'deliverables/manifest.json').write_text(json.dumps(manifest))
        self.report={'all_original_frames_visually_reviewed':False,'protocol':{'object_assistance':0,'failure_resets':False},
                     'validation':[{'video':name,'frames':202 if i<3 else 402,'fps':20} for i,name in enumerate(VIDEOS)]}
        self.write_report()

    def write_report(self):
        (self.root/'result-report.json').write_text(json.dumps(self.report))

    def test_manual_review_stays_required_by_default(self):
        self.assertIn('source review incomplete',problems(self.root,verify_hashes=True))
        self.assertEqual(problems(self.root,verify_hashes=True,require_source_review=False),[])

    def test_diagnostic_mode_still_rejects_modified_artifacts(self):
        (self.root/'deliverables/policy.safetensors').write_bytes(b'corrupted')
        self.assertTrue(problems(self.root,verify_hashes=True,require_source_review=False))

    def test_diagnostic_mode_still_rejects_object_assistance(self):
        self.report['protocol']['object_assistance']=1;self.write_report()
        self.assertIn('unassisted continuous attempt protocol violated',problems(self.root,require_source_review=False))

if __name__=='__main__':unittest.main()
