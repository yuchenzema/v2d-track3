"""Reapply explicit source-image background annotations after mask propagation."""
import json,os,shutil
from pathlib import Path
import cv2
from improvement_common import digest,save,utc


def apply_review(root):
    root=Path(root);rec=root/'reconstruction';path=rec/'source-background-review.json'
    if not path.exists():return None
    review=json.loads(path.read_text());cfg=json.loads((root/'episode-config.json').read_text())
    if review.get('episode')!=cfg['episode']:raise RuntimeError('Background review episode differs from source')
    pending=[]
    for row in review['annotations']:
        frame=row['frame'];body=row['object_id']
        if not (0<=frame<cfg['frames'] and 1<=body<=len(cfg['objects'])):raise RuntimeError('Background review frame/object out of bounds')
        image=rec/f'frames/{frame:06d}.png'
        if digest(image)!=row['source_frame_sha256']:raise RuntimeError('Background review source image hash mismatch')
        target=rec/f'masks/{body}/{frame:06d}.png';mask=cv2.imread(str(target),0)
        if mask is None:raise RuntimeError('Missing background-review source mask')
        pixels=row['background_pixels']
        if not pixels or any(not isinstance(x,int) or not isinstance(y,int) or not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]) for x,y in pixels):
            raise RuntimeError('Background annotation pixel out of bounds')
        pending.append((row,target,mask))
    records=[]
    for row,target,mask in pending:
        before=digest(target);removed=sum(bool(mask[y,x]) for x,y in row['background_pixels'])
        if removed:
            backup=root/f'research/source-background-mask-backups/{before}.png';backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():shutil.copy2(target,backup)
            for x,y in row['background_pixels']:mask[y,x]=0
            from source_cap_masks import writable_mask_directory
            writable_mask_directory(root,target.parent)
            temp=target.with_name(f'{target.stem}.background-{os.getpid()}.png')
            try:
                if not cv2.imwrite(str(temp),mask):raise RuntimeError('Background annotation mask write failed')
                temp.replace(target)
            finally:temp.unlink(missing_ok=True)
        records.append(dict(frame=row['frame'],object_id=row['object_id'],removed_foreground_pixels=removed,
                            previous_mask_sha256=before,current_mask_sha256=digest(target)))
    report=dict(updated_utc=utc(),review_sha256=digest(path),records=records,foreground_added=False,
                source_images_modified=False,geometry_or_pose_used=False,quality_gates_changed=False)
    save(root/'research/source-background-review-applied.json',report);return report
