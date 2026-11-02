"""Verify unchanged original videos and retain the scope of their prior visual review."""
import argparse
import json
from pathlib import Path
from improvement_common import save, utc, digest

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
c=json.loads((r/'episode-config.json').read_text())
old=r.parent/f"episode_{c['episode']:06d}_newton_full_rebuild_20261022"
before=json.loads((old/'research/rebuild-input-provenance.json').read_text())['source_videos']
now=json.loads((r/'research/rebuild-input-provenance.json').read_text())['source_videos']
assert before==now
for item in now:assert digest(item['path'])==item['sha256']
prior=old/('research/full-frame-visual-review/review.json' if c['episode']==19 else 'research/full-source-visual-review/review.json')
review=json.loads(prior.read_text());assert review['source_frames_reviewed']==list(range(c['frames']))
record={'updated_utc':utc(),'method':'Original-video visual review retained only after verifying all three original video hashes; new reconstruction is independently scored on every frame. New policy visual review is pending.',
        'source_frames_reviewed':review['source_frames_reviewed'],'prior_source_review':str(prior),
        'new_reconstruction_manually_verified':False,'new_policy_manually_verified':False,'original_video_hashes_verified':True}
save(r/'research/source-review-verified.json',record)
c.update(all_frames_visually_reviewed=True,visual_review_record='research/source-review-verified.json',visual_review_method=record['method'])
save(r/'episode-config.json',c)
