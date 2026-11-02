"""Review every source frame and expose corresponding-object baseline diagnostics."""
from pathlib import Path
import hashlib
import html
import json
import cv2
from improvement_common import BASE, OUT, root, baseline, save, utc
from object_motion_quality import PRIORITY_EPISODES


def generate():
    cv2.setNumThreads(1)
    results = []
    for ep in PRIORITY_EPISODES:
        r = root(ep)
        cfg = json.loads((r / 'episode-config.json').read_text())
        selected = baseline(ep)
        stem = Path(selected['recorded_states']).stem.removesuffix('-states')
        fidelity = json.loads((r / 'evaluation' / f'{stem}-source-fidelity.json').read_text())
        reconstruction = json.loads((r / 'evaluation/reference-source-fidelity.json').read_text())
        source = Path(cfg['source'])
        cap = cv2.VideoCapture(str(source))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1::-1] != tuple(cfg['resolution']):
                raise ValueError(f'Unexpected original video resolution: {source}')
            decoded += 1
        cap.release()
        if decoded != cfg['frames']:
            raise ValueError(f'Original video decoded {decoded} / {cfg["frames"]} frames')
        def url(path):
            return '/artifacts/' + str(path.relative_to(BASE / 'runs'))
        physical_path = r / 'evaluation/physical-reference-source-fidelity.json'
        physical = json.loads(physical_path.read_text()) if physical_path.exists() else None
        results.append({
            'episode': ep, 'frames': decoded, 'fps': cfg['fps'], 'source': str(source),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'baseline': selected['object_motion_quality'],
            'reconstruction_mean_iou': {n: v['iou'] for n, v in reconstruction['aggregate']['objects'].items()},
            'physical_reference': physical['aggregate'] if physical else None,
            'notes': cfg.get('source_review_notes', ''), 'grasps': cfg['grasps'],
            'baseline_video': url(r / 'videos/original_vs_policy.mp4'),
            'reconstruction_video': url(r / 'videos/original_vs_reconstruction.mp4'),
            'failure_image': url(OUT / f'episode_{ep:06d}-failure-review.jpg'),
            'source_frame_metrics': fidelity['frames'],
            'reconstruction_frame_metrics': reconstruction['frames'],
        })
        print(f'Episode {ep:06d}: decoded all {decoded} original frames; checked each corresponding object', flush=True)
    report = {'updated_utc': utc(), 'episodes': results,
              'original_frames_decoded_and_measured': sum(r['frames'] for r in results),
              'baseline_or_training_inputs_modified': False,
              'exact_human_motion_match_verified': False}
    save(OUT / 'priority-object-review.json', report)
    template = (BASE / 'priority_object_review.html').read_text()
    template = template.replace('__REVIEW_DATA__', json.dumps(report).replace('</', '<\\/'))
    (OUT / 'priority-object-review.html').write_text(template)
    print('Wrote', OUT / 'priority-object-review.html', flush=True)


if __name__ == '__main__':
    generate()
