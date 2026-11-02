"""One public-source stereo smoke test; does not replace reconstruction depths."""
from pathlib import Path
import json
import subprocess
import cv2
import numpy as np
from batch_worker import docker
from improvement_common import BASE,save,utc,digest

ROOT=BASE/'runs/public_episode_000000_pipeline_20261029'
REPO=BASE/'third_party/FoundationStereo'


def run():
    folder=ROOT/'research/foundation-stereo-probe';folder.mkdir(exist_ok=True)
    data=BASE/'datasets/video_to_data_challenge/track_3'
    cal=json.loads((data/'public/meta/camera_calibration.json').read_text());pair=cal['stereo_pairs'][1]
    fit=np.load(ROOT/'reconstruction/camera-frame-calibration.npz');transform=fit['stereo_left_to_right']
    cameras=[cal['cameras'][pair[x]] for x in ['left','right']]
    K=[np.array(c['intrinsic_matrix']) for c in cameras];D=[np.array(c['distortion_coefficients']) for c in cameras]
    R1,R2,P1,P2,*_=cv2.stereoRectify(K[0],D[0],K[1],D[1],(1280,800),transform[:3,:3].copy(),transform[:3,3].copy().reshape(3,1),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    for index,name in enumerate(['left','right']):
        capture=cv2.VideoCapture(str(data/f'public/videos/chunk-000/observation.images.{pair[name]}/episode_000000.mp4'))
        ok,frame=capture.read();capture.release();assert ok
        maps=cv2.initUndistortRectifyMap(K[index],D[index],[R1,R2][index],[P1,P2][index],(1280,800),cv2.CV_32FC1)
        cv2.imwrite(str(folder/f'{name}.png'),cv2.remap(frame,*maps,cv2.INTER_LINEAR))
    (folder/'K.txt').write_text(' '.join(str(v) for v in P1[:3,:3].ravel())+'\n'+str(pair['baseline_m'])+'\n')
    wrapper=folder/'run-probe.py'
    wrapper.write_text('''from pathlib import Path
import json,runpy,sys,torch,timm,numpy as np
load=torch.load
def trusted_nvidia_load(*a,**kw):
 kw.setdefault('weights_only',False);return load(*a,**kw)
torch.load=trusted_nvidia_load
hub=torch.hub.load
def local_dino(repo,model,*a,**kw):
 if repo=='facebookresearch/dinov2':repo='/foundation/dinov2';kw['source']='local'
 return hub(repo,model,*a,**kw)
torch.hub.load=local_dino
# The full released checkpoint supplies all backbone tensors. Avoid fetching
# an unrelated second copy of EdgeNeXt weights before strict checkpoint load.
create=timm.create_model
def checkpoint_backbone(*a,**kw):kw['pretrained']=False;return create(*a,**kw)
timm.create_model=checkpoint_backbone
sys.argv=['run_demo.py','--left_file','/probe/left.png','--right_file','/probe/right.png','--ckpt_dir','/foundation/pretrained_models/23-51-11/model_best_bp2.pth','--out_dir','/probe/output','--scale','.5','--valid_iters','16','--get_pc','0']
result=runpy.run_path('/foundation/scripts/run_demo.py',run_name='__main__')
disp=result['disp'];K=np.array([float(x) for x in Path('/probe/K.txt').read_text().splitlines()[0].split()]).reshape(3,3);baseline=float(Path('/probe/K.txt').read_text().splitlines()[1])
depth=K[0,0]*.5*baseline/np.maximum(disp,1e-5)
np.savez_compressed('/probe/output/metric-depth.npz',depth_m=depth,disparity_px=disp)
Path('/probe/output/smoke-test.json').write_text(json.dumps(dict(shape=list(depth.shape),finite_fraction=float(np.isfinite(depth).mean()),median_depth_m=float(np.median(depth[np.isfinite(depth)])),gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),ground_truth_used=False,used_for_current_reconstruction=False)))
''')
    command=['run','--rm','--gpus','all','--ipc','host','-v',f'{REPO}:/foundation:ro','-v',f'{folder}:/probe','-w','/foundation',
             '-e','PYTHONPATH=/foundation/python_packages:/foundation','-e','XFORMERS_DISABLED=1','v2d_foundation_pose','python','/probe/run-probe.py']
    docker(command,ROOT/'logs/foundation-stereo-probe.log')
    result=json.loads((folder/'output/smoke-test.json').read_text())
    result.update(updated_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  checkpoint_sha256=digest(REPO/'pretrained_models/23-51-11/model_best_bp2.pth'),
                  baseline_m=pair['baseline_m'],source='Public episode 0 frame 0, rectified factory C/B stereo pair')
    save(folder/'installation-validation.json',result)


if __name__=='__main__':run()
