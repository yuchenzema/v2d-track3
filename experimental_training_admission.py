"""Explicitly relaxed source admission for pilots, never final completion."""
import json
import math
from pathlib import Path
from improvement_common import BATCH, digest

POLICY = BATCH/'focused_episodes_20261028/experimental-training-policy.json'


def assess(report, policy):
    reasons=[]
    if policy.get('enabled') is not True:reasons.append('Experimental admission is disabled')
    if report.get('episode') not in policy.get('episodes',[]):reasons.append('Episode outside experimental scope')
    if report.get('all_source_frames_checked') is not True:reasons.append('Incomplete source audit')
    if report.get('unobservable_object_frames')!=0:reasons.append('Unobservable source poses remain')
    if report.get('final_scene_occlusion_rechecked') is not True:reasons.append('Final scene audit missing')
    if report.get('source_time_warping_used') is not False:reasons.append('Source clock not preserved')
    if report.get('physical_projection_applied') is not False:
        if not (policy.get('allow_source_revalidated_physical_candidate') is True
                and report.get('reference_kind')=='source_revalidated_physical_candidate'
                and report.get('experimental_only') is True):
            reasons.append('Source poses physically projected without experimental source revalidation')
    groups={}
    try:
        root=Path(report['run']);trajectory=root/'reconstruction/cad-source-trajectory.npz'
        assert digest(trajectory)==report['trajectory_sha256'],'Stale trajectory'
        frames=Path(report['frames_report'])
        assert digest(frames)==report['frames_report_sha256'],'Stale frame evidence'
        cfg=json.loads((root/'episode-config.json').read_text())
        assert digest(Path(cfg['source']))==report['source_video_sha256'],'Stale source video'
        audit=json.loads(Path(report['final_scene_audit']).read_text())
        assert audit.get('all_object_frames_rechecked') is True,'Incomplete final scene audit'
        assert audit.get('trajectory_sha256')==report['trajectory_sha256'],'Stale final scene trajectory'
        assert audit.get('frames_report_sha256')==report['frames_report_sha256'],'Stale final scene frames'
        rows=json.loads(frames.read_text())
        expected={(f,name) for f in range(cfg['frames']) for name in cfg['objects']}
        assert len(rows)==len(expected)==report['object_frames_checked'],'Incomplete frame count'
        assert {(r['frame'],r['object']) for r in rows}==expected,'Duplicate/missing object frames'
        for row in rows:
            values=[row.get(k) for k in ['iou','centroid_error_px','boundary_error_px']]
            assert all(isinstance(x,(int,float)) and math.isfinite(x) for x in values),'Missing/nonfinite image evidence'
            assert row.get('visible_pixels',0)>=60,'Unobservable object frame'
            sphere=row.get('sphere_error_px')
            assert sphere is None or (isinstance(sphere,(int,float)) and math.isfinite(sphere)),'Invalid sphere evidence'
            good=(values[0]>=policy.get('minimum_iou',.85)
                  and values[1]<=policy.get('maximum_centroid_error_px',5.)
                  and values[2]<=policy.get('maximum_boundary_error_px',5.)
                  and (sphere is None or sphere<=3.))
            groups.setdefault(row['object'],[]).append(good)
    except (OSError,ValueError,KeyError,AssertionError,TypeError) as error:
        reasons.append(str(error))
    fractions={name:sum(good)/len(good) for name,good in groups.items()}
    if not fractions or min(fractions.values())<policy.get('minimum_qualified_fraction',.85):
        reasons.append('Insufficient qualifying frames for at least one object')
    return dict(admitted=not reasons,reasons=reasons,qualified_fraction_by_object=fractions,
                minimum_qualified_fraction=min(fractions.values()) if fractions else 0.,
                experimental_only=True,strict_completion_granted=False)


def require(report, cfg):
    policy=json.loads(POLICY.read_text())
    assert cfg.get('experimental_training') is True,'Explicit experimental configuration required'
    result=assess(report,policy)
    assert result['admitted'],result['reasons']
    return result
