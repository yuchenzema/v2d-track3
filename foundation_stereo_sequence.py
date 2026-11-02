"""Rectified C/B sequence inference with consistency checks and metric RGB splats.

Outputs a separate candidate depth directory. Existing reconstruction inputs are
never overwritten; source-CAD validation must decide whether the change helps.
"""
import argparse,json
from pathlib import Path
from improvement_common import BASE,save,utc
from batch_worker import docker


def run(root,progress_root=None):
    root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text())
    destination=rec/'foundation-stereo-sequence.json'
    if destination.exists():return
    folder=root/'research/foundation-stereo-sequence';folder.mkdir(exist_ok=True)
    repo=BASE/'third_party/FoundationStereo';data=BASE/'datasets/video_to_data_challenge/track_3'
    probe=BASE/'runs/public_episode_000000_pipeline_20261029/research/foundation-stereo-probe'
    assert (probe/'installation-validation.json').is_file(),'FoundationStereo smoke test has not passed'
    calibration=json.loads((data/'public/meta/camera_calibration.json').read_text())
    pair=calibration['stereo_pairs'][1]
    split=cfg.get('dataset_split','evaluation')
    job=dict(frames=cfg['frames'],calibration=calibration,split=split,sequence=cfg['sequence'],pair=pair,episode=cfg['episode'],object_count=len(cfg['objects']))
    save(folder/'job.json',job)
    script=folder/'run-sequence.py'
    script.write_text('''import json,runpy,time
from pathlib import Path
import cv2,numpy as np,torch
from core.utils.utils import InputPadder
globals=runpy.run_path('/probe/run-probe.py');model=globals['result']['model'];model.eval()
root=Path('/episode');rec=root/'reconstruction';job=json.loads((root/'research/foundation-stereo-sequence/job.json').read_text());pair=job['pair'];cal=job['calibration'];fit=np.load(rec/'camera-frame-calibration.npz')
E=fit['stereo_left_to_right'];cameras=[cal['cameras'][pair[x]] for x in ['left','right']];K=[np.array(c['intrinsic_matrix']) for c in cameras];D=[np.array(c['distortion_coefficients']) for c in cameras]
R1,R2,P1,P2,*_=cv2.stereoRectify(K[0],D[0],K[1],D[1],(1280,800),E[:3,:3].copy(),E[:3,3].copy().reshape(3,1),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
maps=[cv2.initUndistortRectifyMap(K[j],D[j],[R1,R2][j],[P1,P2][j],(1280,800),cv2.CV_32FC1) for j in range(2)]
caps=[cv2.VideoCapture(f"/dataset/{job['split']}/videos/chunk-000/observation.images.{pair[x]}/{job['sequence']}.mp4") for x in ['left','right']]
out=rec/'depth_foundation_stereo';out.mkdir(exist_ok=True);yy,xx=np.indices((400,640),dtype=np.float32);intr=fit['K_rgb'];reports=[]
def predict(left,right):
 a=torch.from_numpy(left.copy()).cuda().float()[None].permute(0,3,1,2);b=torch.from_numpy(right.copy()).cuda().float()[None].permute(0,3,1,2)
 pad=InputPadder(a.shape,divis_by=32,force_square=False);a,b=pad.pad(a,b)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):d=model.forward(a,b,iters=16,test_mode=True)
 return pad.unpad(d.float()).cpu().numpy().reshape(400,640)
for frame in range(job['frames']):
 images=[]
 for j,cap in enumerate(caps):
  ok,image=cap.read();assert ok
  image=cv2.remap(image,*maps[j],cv2.INTER_LINEAR);images.append(cv2.cvtColor(cv2.resize(image,(640,400)),cv2.COLOR_BGR2RGB))
 dl=predict(*images);dr=np.fliplr(predict(np.fliplr(images[1]),np.fliplr(images[0])))
 right=cv2.remap(dr,xx-dl,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-999)
 z=P1[0,0]*.5*pair['baseline_m']/np.maximum(dl,1e-5)
 valid=(dl>2)&(np.abs(dl-right)<1.)&(z>.2)&(z<5.)&np.isfinite(z)
 pc=np.stack([(xx-P1[0,2]*.5)/(P1[0,0]*.5)*z,(yy-P1[1,2]*.5)/(P1[1,1]*.5)*z,z],axis=-1)[valid]@R1
 M=fit['stereo_left_to_rgb_per_frame'][frame];pc=pc@M[:3,:3].T+M[:3,3]
 uv=np.rint(pc[:,:2]/pc[:,2:]*[intr[0,0],intr[1,1]]+[intr[0,2],intr[1,2]]).astype(int)
 keep=(uv[:,0]>=0)&(uv[:,0]<1024)&(uv[:,1]>=0)&(uv[:,1]<768)&(pc[:,2]>.1);uv=uv[keep];pc=pc[keep]
 depth=np.full(768*1024,np.inf,np.float32);np.minimum.at(depth,uv[:,1]*1024+uv[:,0],pc[:,2]);depth=depth.reshape(768,1024);good=np.isfinite(depth)
 encoded=np.zeros((768,1024),np.uint16);encoded[good]=np.clip(65535/(depth[good]+1),0,65535).astype(np.uint16);cv2.imwrite(str(out/f'{frame:06d}.png'),encoded)
 reports.append(dict(frame=frame,valid_stereo_fraction=float(valid.mean()),rgb_coverage=float(good.mean())))
 if frame%10==0:
  progress=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),frame=frame+1,total_frames=job['frames'],method='FoundationStereo metric sequence, 16 iterations, left/right consistency')
  temp=root/'research/foundation-stereo-progress.tmp';temp.write_text(json.dumps(progress));temp.replace(root/'research/foundation-stereo-progress.json')
  if Path('/cad/research').exists():
   progress.update(episode=job['episode'],phase='FoundationStereo full-sequence metric depth before CAD fitting',object_frames_processed=0,total_object_frames=job['frames']*job['object_count'],policy_training_started=False)
   temp=Path('/cad/research/cad-progress.tmp');temp.write_text(json.dumps(progress));temp.replace('/cad/research/cad-progress.json')
for cap in caps:cap.release()
temp=rec/'foundation-stereo-sequence.tmp';temp.write_text(json.dumps(dict(frames=job['frames'],baseline_m=pair['baseline_m'],source_left=pair['left'],source_right=pair['right'],method='FoundationStereo 640x400, bidirectional disparity consistency, metric baseline, per-frame RGB registration and z-buffer; no spread across depth edges',per_frame=reports,source_inputs_overwritten=False,ground_truth_used=False)));temp.replace(rec/'foundation-stereo-sequence.json')
''')
    docker(['run','--rm','--gpus','all','--ipc','host',*(['-v',f'{Path(progress_root)}:/cad'] if progress_root else []),'-v',f'{repo}:/foundation:ro','-v',f'{root}:/episode',
            '-v',f'{data}:/dataset:ro','-v',f'{probe}:/probe','-w','/foundation',
            '-e','PYTHONPATH=/foundation/python_packages:/foundation','-e','XFORMERS_DISABLED=1',
            'v2d_foundation_pose','python','/episode/research/foundation-stereo-sequence/run-sequence.py'],root/'logs/foundation-stereo-sequence.log')
    assert destination.exists()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();run(a.run)
