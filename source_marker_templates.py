"""Distinguish source-reviewed fiducials from generic spherical CAD features."""
import copy
import json
from pathlib import Path

from improvement_common import digest,save,utc


def load(path,review_path=None,report_path=None):
    path=Path(path);data=json.loads(path.read_text());templates=copy.deepcopy(data['objects'])
    manifest=dict(updated_utc=utc(),template_file=str(path),template_sha256=digest(path),
                  official_marker_annotations=False,ground_truth_used=False,objects={})
    if review_path and Path(review_path).is_file():
        review_path=Path(review_path);review=json.loads(review_path.read_text())
        if review['template_sha256']!=digest(path):raise ValueError('Reviewed marker feature IDs belong to another template version')
        if not review.get('source_images'):raise ValueError('Marker-feature review needs original source image evidence')
        for evidence in review['source_images']:
            if digest(Path(evidence['path']))!=evidence['sha256']:raise ValueError('Source image for marker-feature review changed')
        manifest['review']=dict(path=str(review_path),sha256=digest(review_path))
        for name,decision in review['objects'].items():
            template=templates[name];mesh=Path(template['mesh'])
            if decision['mesh_sha256']!=template['mesh_sha256'] or digest(mesh)!=decision['mesh_sha256']:
                raise ValueError('Reviewed marker feature IDs belong to another CAD mesh')
            markers=template['markers'];ids=decision['retained_marker_ids']
            if not ids or len(ids)!=len(set(ids)) or any(not isinstance(i,int) or not 0<=i<len(markers) for i in ids):
                raise ValueError('Invalid reviewed fiducial IDs')
            selected=[]
            for old_id in ids:
                marker=copy.deepcopy(markers[old_id]);marker['source_marker_id']=old_id;marker['marker_id']=len(selected);selected.append(marker)
            template['markers']=selected
            manifest['objects'][name]=dict(original_curvature_candidates=len(markers),used_fiducials=len(selected),
                retained_source_marker_ids=ids,rejected_source_marker_ids=[i for i in range(len(markers)) if i not in ids],
                reason=decision['reason'],mesh_sha256=decision['mesh_sha256'])
    if report_path:save(Path(report_path),manifest)
    return templates
