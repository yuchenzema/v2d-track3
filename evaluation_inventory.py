"""Audit all twenty HF evaluation episodes, including the earlier standalone1/5 runs."""
import json,re,subprocess,urllib.request
from pathlib import Path
import cv2,numpy as np
from improvement_common import BASE,BATCH,digest,save,utc

OUT=BATCH/'all20_evaluation'
LEGACY={1:BASE/'runs/episode_000001_newton_v2_20261014',5:BASE/'runs/episode_000005_newton_20261015'}


def legacy_record(ep):
    path=OUT/f'episode_{ep:06d}-release.json'
    return json.loads(path.read_text())


def prepare_legacy():
    OUT.mkdir(exist_ok=True)
    root=LEGACY[1];recording=BASE/'runs/episode_000001_newton_v5_20261014'
    for name,filter in [('comparison','crop=1280:480:0:0'),('rollout','crop=640:480:640:0')]:
        path=OUT/f'episode_000001-{name}.mp4'
        if not path.exists():
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(recording/'videos/before_after_comparison.mp4'),
                '-vf',filter,'-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],check=True)
    for ep in (1,5):
        root=LEGACY[ep];report=json.loads((root/'result-report.json').read_text())
        if ep==1:
            files={'comparison':OUT/'episode_000001-comparison.mp4','rollout':OUT/'episode_000001-rollout.mp4',
                   'scene':root/'videos/trained_policy_rollout.mp4','report':root/'result-report.json'}
            state=recording/'evaluation/baseline-policy-states.npz';checkpoint=root/'flash_chord/policy_250003456.safetensors'
            quality={'all_internal_frame_checks_passed':False,'full_human_video_match_verified':False,
                     'completion_rate_against_video_estimate':report['evaluation_success_rate']}
            steps=report['training']['total_environment_steps'];frames=report['source_frames']
            objects=['white_pot','white_pot_lid'];selected=None
        else:
            files={key:root/'videos'/value for key,value in [('comparison','original_vs_policy.mp4'),('rollout','policy_rollout_source_timed.mp4'),('scene','trained_policy_rollout.mp4')]}
            files['report']=root/'result-report.json';selected=json.loads((root/'final-controller-selection.json').read_text())['selected']
            state=root/selected['recorded_states'];checkpoint=root/selected['checkpoint'].removeprefix('/episode/')
            quality={**report['quality_assessment'],'all_internal_frame_checks_passed':False,
                     'completion_rate_against_video_estimate':report['quality_assessment']['native_reference_completion_rates'][0]}
            steps=selected['environment_steps'];frames=report['all_original_rgb_frames_reviewed'];objects=['white_pot','wooden_spoon']
        entries={key:{'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size} for key,path in files.items()}
        version=digest(checkpoint)[:8]+entries['comparison']['sha256'][:8]
        save(OUT/f'episode_{ep:06d}-release.json',{'episode':ep,'run':str(root),'recording_root':str(recording if ep==1 else root),
             'version':version,'validated_utc':utc(),'objects':objects,'source_frames':frames,'environment_steps':steps,
             'quality':quality,'files':entries,'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
             'recorded_states':str(state),'recorded_states_sha256':digest(state),'selected':selected,
             'selection_reason':'Preserve the episode1 policy explicitly preferred by the user' if ep==1 else 'Preserve the completed episode5 selected policy'})


def audit():
    from video_catalog import catalog
    from batch_final_audit import audit as audit_run
    url='https://huggingface.co/api/datasets/nvidia/video_to_data_challenge/tree/main/track_3/evaluation?recursive=true&limit=1000'
    with urllib.request.urlopen(url,timeout=45) as response:inventory=json.load(response)
    videos=[x for x in inventory if x['type']=='file' and x['path'].endswith('.mp4')]
    episodes=sorted({int(re.search(r'episode_(\d+)\.mp4$',x['path'])[1]) for x in videos})
    assert len(episodes)==20 and len(videos)==60
    source_checks=[]
    for entry in videos:
        path=BASE/'datasets/video_to_data_challenge'/entry['path'];sha=digest(path)
        assert sha==entry['lfs']['oid'],f'HF source video differs: {path}'
        source_checks.append({'path':entry['path'],'sha256':sha,'bytes':path.stat().st_size})
    rows={}
    for ep in episodes:
        if ep in LEGACY:
            release=legacy_record(ep);state_path=Path(release['recorded_states']);root=Path(release['run'])
            assert digest(state_path)==release['recorded_states_sha256']
            assert digest(Path(release['checkpoint']))==release['checkpoint_sha256']
            report=json.loads((root/'result-report.json').read_text());status=json.loads((root/'run-status.json').read_text())
            assert status['train' if ep==1 else 'training']['status']=='completed'
            for entry in release['files'].values():assert digest(Path(entry['path']))==entry['sha256']
            frames=release['source_frames'];files=release['files'];quality=release['quality']
        else:
            baseline=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015'
            baseline_check=audit_run(baseline);assert baseline_check['integrity_passed'],baseline_check
            release=catalog(ep,baseline,True)['latest'];root=Path(release['run']);selected=release['selected']
            state_path=root/selected['recorded_states'];frames=json.loads((root/'episode-config.json').read_text())['frames']
            files={role:{'path':str(BASE/'runs'/link.split('/artifacts/')[1].split('?')[0])} for role,link in release['links'].items() if role in ('comparison','rollout','scene')}
            quality=release['quality']
        with np.load(state_path,allow_pickle=False) as state:
            assert len(state['body_poses'])==2*(frames-1) and np.isfinite(state['body_poses']).all()
            assert np.all(state['object_assistance']==0) and np.array_equal(state['reference_frames'],np.arange(2*(frames-1)))
        validations={}
        for role in ('comparison','rollout','scene'):
            path=Path(files[role]['path']);cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.size;count+=1
            cap.release();expected=2*(frames-1) if role=='scene' else frames
            assert count==expected and abs(fps-20)<.01,(ep,role,count,expected,fps)
            validations[role]={'path':str(path),'sha256':digest(path),'frames':count,'fps':fps}
        rows[str(ep)]={'episode':ep,'run':str(root),'version':release['version'],'source_frames':frames,
             'reconstruction_complete':True,'training_complete':True,'rollout_complete_and_integrity_verified':True,
             'videos':validations,'quality':quality,'full_human_video_match_verified':False}
        print(ep,'reconstruction, training and full-frame rollout integrity verified',flush=True)
    save(OUT/'all20-episode-audit.json',{'updated_utc':utc(),'hf_evaluation_url':url,'expected_episodes':episodes,
         'expected_episode_count':20,'source_video_hashes':source_checks,'episodes':rows,
         'all_20_pipeline_deliverables_complete_and_verified':len(rows)==20,
         'exact_human_imitation_verified':False,'note':'Pipeline completion and integrity do not certify imitation quality or official challenge success.'})


if __name__=='__main__':
    import sys
    prepare_legacy()
    if '--legacy-only' not in sys.argv:audit()
