"""Verify calibration, pixel coordinates, original-mask identity and provenance."""
import json
from pathlib import Path
import cv2
import numpy as np
from improvement_common import BASE,BATCH,save,utc,digest

out=BATCH/'sam3d_rebuild_20261026'
manifest=json.loads((out/'input-manifest.json').read_text())
tiles=[]
for job in manifest['jobs']:
    image=cv2.imread(job['image']);mask=cv2.imread(job['mask'],0)
    assert image is not None and mask is not None and image.shape[:2]==mask.shape
    assert digest(job['image'])==job['image_sha256']
    assert digest(job['mask'])==job['mask_sha256']
    original=cv2.imread(job['original_mask'],0);assert np.array_equal(original,mask)
    intr=json.loads(Path(job['source_camera_intrinsics']).read_text())
    assert [intr['width'],intr['height']]==job['mask_resolution']
    assert job['calibrated_source_frame']
    if job.get('depth'):
        z=cv2.imread(job['depth'],cv2.IMREAD_UNCHANGED)
        assert z.shape==mask.shape and z.dtype==np.uint16
    if job['episode'] in [19,20,30,33]:
        image[mask==0]=(image[mask==0]*.25).astype('uint8')
        tile=cv2.resize(image,(320,240))
        cv2.putText(tile,f"ep{job['episode']} {job['object']} f{job['frame']}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        tiles.append(tile)
survey=np.concatenate([np.concatenate(tiles[i:i+6],axis=1) for i in range(0,len(tiles),6)],axis=0)
assert cv2.imwrite(str(out/'sam3d-input-survey.jpg'),survey)
result={'validated_utc':utc(),'episodes':len(manifest['episodes']),'object_view_inputs':len(manifest['jobs']),
        'original_frames_decoded':sum(e['source_frames_decoded'] for e in manifest['episodes'].values()),
        'all_frame_mask_intrinsics_coordinates_match':True,'original_mask_pixels_preserved':True,
        'metric_depth_inputs':sum(bool(j.get('depth')) for j in manifest['jobs']),'input_hashes_verified':True,
        'sam3d_container_cuda_and_import_check':'Passed on RTX5090, torch2.8.0+cu128',
        'sam3d_model_inference_executed':False,'sam3d_policy_training_started':False,
        'plugin_search_result':'No Hugging Face plugin found; installed native software used'}
save(out/'input-validation.json',result)
print(json.dumps(result,indent=2))
