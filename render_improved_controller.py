"""Render an improved genuine policy attempt alongside its untouched baseline."""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from batch_native import docker,containers
from improvement_common import BASE,root,digest,save


def render(ep,selected,destination,root_override=None,container_override=None):
    r=Path(root_override) if root_override else root(ep);cfg=json.loads((r/'episode-config.json').read_text());T=cfg['frames'];N=2*(T-1)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    container=container_override or containers(r,ep)[0];state=r/selected['recorded_states'];checkpoint=selected['checkpoint']
    relative='/episode/'+str(destination.relative_to(r));state_relative='/episode/'+str(state.relative_to(r))
    solver_overrides=[f'sim.{key}={value}' for key,value in selected.get('solver_capacity',{}).items()]
    solver_overrides += [f'scene.{key}={str(value).lower()}' for key,value in selected.get('scene_parameters',{}).items()]
    render_reference=[f"task.parquet='/episode/human_motion_data/{selected['inference_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",f"sim.substeps={selected['physics_substeps']}"]
    common=['viewer.backend=mp4','viewer.width=1024','viewer.height=768',f'viewer.num_frames={N}',
            'viewer.video_fps=20','viewer.realtime=false','markers.enabled=false']
    for script,name,extras in [('render_source_camera.py','policy_source_camera',[]),
                               ('render_recorded_oblique.py','policy_3d',['viewer.width=1280','viewer.height=720'])]:
        path=destination/f'{name}.mp4'
        if not path.exists():
            docker(['exec',container,'env',f'V2D_REPLAY_STATE_FILE={state_relative}',
                    f'V2D_STATE_OUTPUT={relative}/source-rendered-states.npz','python','-u',f'/episode/scripts/{script}',
                    f'evaluation.checkpoint={checkpoint}',*render_reference,*solver_overrides,*common,*extras,f'viewer.output_path={relative}/{name}.mp4',
                    f'hydra.run.dir=/episode/hydra/improved-{destination.name}-{name}'],destination/f'{name}.log')
    with np.load(state,allow_pickle=False) as a,np.load(destination/'source-rendered-states.npz',allow_pickle=False) as b:
        assert np.array_equal(a['body_poses'],b['body_poses'])
        assert np.all(a['object_assistance']==0)
    calibration=json.loads((BASE/'datasets/video_to_data_challenge/track_3/public/meta/camera_calibration.json').read_text())['cameras']['ego_cam_a']
    width,height=calibration['resolution'];K=np.array(calibration['intrinsic_matrix']);D=np.array(calibration['distortion_coefficients'])
    newK,_=cv2.getOptimalNewCameraMatrix(K,D,(width,height),.7,(width,height));yy,xx=np.indices((768,1024),dtype=np.float32)
    raw=np.stack([width-1-((xx+.5)*width/1024-.5),height-1-((yy+.5)*height/768-.5)],axis=-1)
    und=cv2.undistortPoints(raw.reshape(-1,1,2),K,D,P=newK).reshape(768,1024,2)
    mx=((width-1-und[:,:,0])*1024/width).astype(np.float32);my=((height-1-und[:,:,1])*768/height).astype(np.float32)
    source=cv2.VideoCapture(cfg['source']);before=cv2.VideoCapture(str(root(ep)/'videos/policy_rollout_source_timed.mp4'))
    after=cv2.VideoCapture(str(destination/'policy_source_camera.mp4'))
    assert all(cap.isOpened() for cap in [source,before,after])
    writers={name:cv2.VideoWriter(str(destination/f'{name}.temporary.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,size)
             for name,size in [('original_vs_improved_policy',(1280,480)),('original_before_after',(1920,480)),
                               ('rollout_source_timed',(1024,768))]}
    assert all(writer.isOpened() for writer in writers.values());index=-1
    for frame in range(T):
        ok,human=source.read();assert ok
        ok,previous=before.read();assert ok
        target=int(np.rint(frame*(N-1)/(T-1)))
        for _ in range(target-index):ok,current=after.read();assert ok
        index=target;current=cv2.remap(current,mx,my,cv2.INTER_LINEAR)
        writers['rollout_source_timed'].write(current)
        panels=[cv2.resize(im,(640,480)) for im in [human,previous,current]]
        for panel,label in zip(panels,['Original human','Preserved baseline','Trained policy: timed wrist impedance' if selected.get('timed_wrist_impedance_profile') else 'Trained policy: declared grip correction' if selected.get('grip_bias_profile') else 'Trained policy: fixed learned-actor combination' if selected.get('actor_combination') else 'Trained policy: declared wrist calibration' if (selected.get('residual_gain_profile') or {}).get('wrist_position_bias_m') else 'Trained policy: declared residual gains' if selected.get('residual_gain_profile') else f'Trained policy: checkpoint {selected["environment_steps"]}']):
            cv2.rectangle(panel,(0,0),(640,35),(20,20,20),-1)
            cv2.putText(panel,f'{label} | source frame {frame}',(8,23),0,.48,(255,255,255),1,cv2.LINE_AA)
        writers['original_vs_improved_policy'].write(np.concatenate([panels[0],panels[2]],axis=1))
        writers['original_before_after'].write(np.concatenate(panels,axis=1))
    for cap in [source,before,after]:cap.release()
    for name,writer in writers.items():
        writer.release();temporary=destination/f'{name}.temporary.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-c:v','libx264',
                        '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination/f'{name}.mp4')],check=True)
        temporary.unlink()
    validated=[]
    for name,length in [('policy_source_camera',N),('policy_3d',N),('original_vs_improved_policy',T),('original_before_after',T),('rollout_source_timed',T)]:
        path=destination/f'{name}.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==length and abs(fps-20)<.01,(name,count,length,fps)
        validated.append({'path':str(path),'frames':count,'fps':fps,'sha256':digest(path)})
    reference=selected['inference_reference'];profile=selected.get('residual_gain_profile');combination=selected.get('actor_combination');grip=selected.get('grip_bias_profile')
    timed=selected.get('timed_wrist_impedance_profile')
    controller_env=([f'V2D_TIMED_WRIST_IMPEDANCE_PROFILE={json.dumps(timed,sort_keys=True)}'] if timed else [f'V2D_GRIP_BIAS={json.dumps(grip,sort_keys=True)}'] if grip else
                    [f'V2D_ACTOR_COMBINATION={json.dumps(combination,sort_keys=True)}'] if combination else
                    [f'V2D_RESIDUAL_PROFILE={json.dumps(profile,sort_keys=True)}'] if profile else [])
    if timed and combination:controller_env.append(f'V2D_ACTOR_COMBINATION={json.dumps(combination,sort_keys=True)}')
    if combination and profile:controller_env.append(f'V2D_RESIDUAL_PROFILE={json.dumps(profile,sort_keys=True)}')
    if selected.get('robot_tracking_feedback_profile'):
        assert selected.get('recording_protocol'),'Robot feedback reproduction requires the native cohort recorder'
        controller_env.append('V2D_ROBOT_TRACKING_FEEDBACK='+json.dumps(selected['robot_tracking_feedback_profile'],sort_keys=True))
    for key,var in [('robot_wrist_calibration_profile','V2D_ROBOT_WRIST_CALIBRATION'),('brush_bristle_compliance_profile','V2D_BRISTLE_COMPLIANCE')]:
        if selected.get(key):
            assert selected.get('recording_protocol'),'Calibrated dynamics require the native cohort recorder'
            controller_env.append(var+'='+json.dumps(selected[key],sort_keys=True))
    record_script=('record_actor_timed_impedance.py' if timed and combination else 'record_timed_impedance.py' if timed else 'record_grip_bias.py' if grip else 'record_actor_combination.py' if combination else
                   'record_residual_profile.py' if profile else 'record_hybrid_controller.py' if selected.get('robot_release_controller_spec') else
                   'record_velocity_controller.py' if selected['wrist_reference_velocity_gain'] else 'record_policy_states.py')
    if combination and profile:record_script='record_actor_calibrated.py'
    command=['exec',container,'env',*controller_env,
             f'V2D_WRIST_VELOCITY_GAIN={selected["wrist_reference_velocity_gain"]}',
             *([f'V2D_RELEASE_SPEC={json.dumps(selected["robot_release_controller_spec"])}'] if selected.get('robot_release_controller_spec') else []),
             f'V2D_STATE_OUTPUT={relative}/reproduced-policy-states.npz','python','-u','/episode/scripts/'+record_script,
             f'evaluation.checkpoint={checkpoint}',f'sim.substeps={selected["physics_substeps"]}',*solver_overrides,
             f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",
             'viewer.backend=none',f'hydra.run.dir=/episode/hydra/improved-{destination.name}-reproduce']
    recording=selected.get('recording_protocol')
    if recording:
        assert recording['recorded_world_index']==0
        world_count=int(recording['simulation_world_count'])
        command=['exec',container,'env',*controller_env,
            f'V2D_RECORD_COHORT_SIZE={world_count}',f'V2D_WRIST_VELOCITY_GAIN={selected["wrist_reference_velocity_gain"]}',
            *([f'V2D_RELEASE_SPEC={json.dumps(selected["robot_release_controller_spec"])}'] if selected.get('robot_release_controller_spec') else []),
            f'V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{reference}/processed/sequence_id={cfg["sequence"]}/robot_name=sharpa_wave',
            f'V2D_SIM_SUBSTEPS={selected["physics_substeps"]}',*[f'V2D_SIM_{key.upper()}={value}' for key,value in selected.get('solver_capacity',{}).items()],
            *([f'V2D_SCENE_DECOMPOSE_OBJECTS={str(selected["scene_parameters"]["decompose_objects"]).lower()}'] if selected.get('scene_parameters') else []),
            f'V2D_STATE_OUTPUT={relative}/reproduced-policy-states.npz','python','-u','/episode/scripts/record_native_cohort.py',
            f'evaluation.checkpoint={checkpoint}',f'evaluation.world_count={world_count}','logging.mode=disabled',
            f'evaluation.metrics_output={relative}/reproduced-cohort-metrics.json',f'hydra.run.dir=/episode/hydra/improved-{destination.name}-reproduce']
    reproduce=destination/'reproduce_improved_policy.py'
    reproduce.write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,'+repr(str(BASE))+')\nfrom batch_native import docker\n'
                         +'docker('+repr(command)+',Path('+repr(str(destination/'reproduce.log'))+'))\n')
    protocol={'kind':'Learned actor with source-timed robot wrist impedance; exact replay of one continuous unassisted Newton rollout' if timed else 'Learned actor with declared finger correction; exact replay of one continuous unassisted Newton rollout' if grip else 'Fixed learned-actor combination; exact replay of one continuous unassisted Newton rollout' if combination else 'Learned actor with declared robot command calibration; exact replay of one continuous unassisted Newton rollout' if profile else 'Learned actor checkpoint; exact replay of one continuous unassisted Newton rollout',
              'evaluation_run':str(r),
              'selected':selected,'recorded_states':str(state),'recorded_states_sha256':digest(state),
              'policy_checkpoint_sha256':digest(r/checkpoint.removeprefix('/episode/')),
              'reference_parquet_sha256':digest(next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'))),
              'source_frame_to_control_frame':'round(i*(N-1)/(T-1)); uniform source timestamps; no DTW',
              'motion_speed':.5,'object_assistance':0.,'failure_resets':False,'original_videos_unchanged':True,
              'both_views_render_identical_physical_poses':True,'validated_videos':validated,
              'reproduce_script':str(reproduce),'full_human_motion_match_verified':False}
    if selected.get('robot_tracking_feedback_profile'):
        protocol['kind']='Learned actor with bounded robot wrist tracking feedback; exact replay of one continuous unassisted Newton rollout'
    save(destination/'manifest.json',protocol)
    return protocol
