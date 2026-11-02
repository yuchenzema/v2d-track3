"""Read current submission states without uploading or spending submission quota."""
import importlib.util
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from improvement_common import BASE, BATCH, save, utc


def refresh():
    spec = importlib.util.spec_from_file_location('v2d_uploader', BASE/'submission_kit_20261102/submit/v2d_submit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    api = module.build_api()
    slugs = json.loads((BASE/'submission_kit_20261102/config/tracks.json').read_text())['tracks']['track_3']['competitions']
    slugs = dict(slugs, cd_o_cm='v2d-challenge-track3-cd-o')

    def read(item):
        key, slug = item
        try:
            submissions = api.competition_submissions(slug, page_size=100) or []
            return key, dict(slug=slug, checked_utc=utc(), submissions=[dict(
                ref=row.ref, state=str(row.status).rsplit('.', 1)[-1],
                public_score=row.public_score, error=row.error_description,
                date=str(row.date)) for row in submissions])
        except Exception as error:
            return key, dict(slug=slug, checked_utc=utc(), read_error=str(error))

    with ThreadPoolExecutor(max_workers=3) as pool:
        metrics = dict(pool.map(read, slugs.items()))
    data = dict(checked_utc=utc(), metrics=metrics, read_only=True, uploads_performed=0)
    save(BATCH/'kaggle-live-status.json', data)
    for key, row in metrics.items():
        submissions = row.get('submissions', [])
        print(key, 'valid', sum(s['state']=='COMPLETE' for s in submissions),
              'invalidated', sum('invalidated' in (s['error'] or '') for s in submissions),
              row.get('read_error', ''), flush=True)
    return data


if __name__ == '__main__':
    refresh()
