"""Audit shortlisted grasp diagnostics against every original source frame."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse
import importlib.util
import json
import cv2
import numpy as np
import trimesh
from improvement_common import BASE,BATCH,baseline,root,save,utc,digest
from object_motion_quality import summarize,nonregressing_objects


def assess(ep,kind='dynamic',label_override=None):
    cv2.setNumThreads(1)
    run=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((run/'episode-config.json').read_text())
    label='grasp-approach-search' if kind=='approach' else 'dynamic-grasp-search'
    if label_override:
        assert label_override.replace('-','').isalnum()
        label=label_override
    details=json.loads((run/f'evaluation/{label}.json').read_text())
    states=np.load(run/f'evaluation/{label}.npz')
    if kind=='approach':details['searches']={'all_objects':{'selected_candidates':details['selected_candidates'],'trials':details['trials']}}
    names=states['body_names'].tolist();ids=[names.index(f'track3_{n}/object') for n in cfg['objects']]
    original=baseline(ep)
    stem=__import__('pathlib').Path(original['recorded_states']).stem.removesuffix('-states')
    phases=json.loads((root(ep)/f'evaluation/{stem}-source-fidelity.json').read_text())['phases']
    spec=importlib.util.spec_from_file_location('dynamic_grasp_geometry',run/'scripts/source_geometry.py')
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    motion=np.load(run/'reconstruction/scene_motion_physical.npz')
    measure=np.load(run/'reconstruction/hand_measurements.npz')
    meshes=[trimesh.load(run/f'reconstruction/mesh_priors/{name}.ply',process=False) for name in cfg['objects']]
    sample=np.rint(np.arange(cfg['frames'])*(details['full_native_control_frames']-1)/(cfg['frames']-1)).astype(int)
    records=[]
    for target,search in details['searches'].items():
        for candidate in search['selected_candidates']:
            row=search['trials'][candidate]
            key=f'candidate_{candidate}_body_poses' if kind=='approach' else f'{target}_candidate_{candidate}_body_poses'
            records.append({'target_object':target,**row,'frames':[],'poses':states[key]})
    for frame in range(cfg['frames']):
        camera=np.linalg.inv(motion['camera_to_world'][frame]);visible=~geometry.human_occlusion(measure,frame)
        masks=[cv2.resize(cv2.imread(str(run/f'reconstruction/masks/{b+1}/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(len(cfg['objects']))]
        for row in records:
            objects={}
            for b,name in enumerate(cfg['objects']):
                valid=visible.copy()
                for other,mask in enumerate(masks):
                    if other!=b:valid&=~mask
                render=geometry.silhouette(meshes[b].vertices,camera@geometry.matrix(row['poses'][sample[frame],ids[b]]))
                objects[name]=geometry.overlap(render,masks[b],valid)
            row['frames'].append({'source_frame':frame,'objects':objects})
    for row in records:
        tracking={'passed_all_frames':False,'per_body':row['objects']}
        quality=summarize(cfg,{'frames':row['frames'],'phases':phases},tracking)
        row.pop('poses');row['object_motion_quality']=quality
        row['passes_per_object_nonregression_vs_baseline']=nonregressing_objects({'object_motion_quality':quality},original)
    result={'episode':ep,'updated_utc':utc(),'diagnostic_only':True,'policy_inference':False,
            'source_frames_checked_per_candidate':cfg['frames'],'source_sha256':digest(cfg['source']),
            'states_sha256':digest(run/f'evaluation/{label}.npz'),'candidates':records}
    save(run/f'evaluation/{label}-source-review.json',result)
    for target in details['searches']:
        selected=sorted([r for r in records if r['target_object']==target],key=lambda r:r['object_motion_quality']['score'],reverse=True)
        print(ep,target,[{'candidate':r['candidate'],'score':round(r['object_motion_quality']['score'],3),
            'objects':{n:round(v['mean_iou'],3) for n,v in r['object_motion_quality']['objects'].items()},
            'per_object_nonregression':r['passes_per_object_nonregression_vs_baseline']} for r in selected[:3]],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',nargs='+',type=int)
    parser.add_argument('--kind',choices=['dynamic','approach'],default='dynamic')
    parser.add_argument('--label');args=parser.parse_args()
    for episode in args.episodes:assess(episode,args.kind,args.label)
