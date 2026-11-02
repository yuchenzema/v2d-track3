"""Independently test early and final grasp-stage checkpoints, never publish them as full rollouts."""
import fcntl,json,struct,time,traceback
from improvement_common import BASE,BATCH,save,utc
from evaluate_grasp_prefix import evaluate

OUT=BATCH/'improvements_20261020_grasp'
TARGETS=[(19,'grasp_curriculum_20261020'),(20,'backward_grasp_20261020'),
         (30,'grasp_curriculum_20261020'),(32,'backward_grasp_20261020'),(33,'backward_grasp_20261020')]

def metadata(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];assert n<1_000_000
        return json.loads(f.read(n))['__metadata__']

if __name__=='__main__':
    with (OUT/'prefix-checkpoint-watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result_path=OUT/'prefix-checkpoint-watch-status.json'
        state=json.loads(result_path.read_text()) if result_path.exists() else {'checks':{},'errors':{}}
        while True:
            all_finished=True
            for ep,variant in TARGETS:
                root=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
                if not (root/'grasp_pretraining/TRAINING_COMPLETED').exists():all_finished=False
                for stage in range(4):
                    folder=root/f'grasp_pretraining/stage_{stage}'
                    paths=sorted(folder.glob('policy_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]))
                    strict=[]
                    for p in paths:
                        if not p.with_name(p.name.replace('policy_','state_')).exists() or time.time()-p.stat().st_mtime<10:continue
                        try:
                            m=metadata(p);cfg=json.loads(m['user.resolved_config_json'])
                            if cfg['env']['reset']['voc_decay_steps']==0 and cfg['env']['voc']['max_force']==0. and cfg['env']['voc']['max_torque']==0.:strict.append(p)
                        except (ValueError,KeyError,OSError):continue
                    if not strict:continue
                    selected=[strict[0]]
                    if (folder/'TRAINING_COMPLETED').exists() and strict[-1]!=strict[0]:selected.append(strict[-1])
                    for path in selected:
                        step=int(path.stem.split('_')[-1]);key=f'{ep}:{variant}:{stage}:{step}'
                        if key in state['checks']:continue
                        previous_error=state['errors'].get(key,{})
                        if previous_error.get('attempts',0)>=2:continue
                        try:
                            report=evaluate(ep,step,stage,variant)
                            state['checks'][key]={'episode':ep,'run':str(root),'backward_stage':stage,'checkpoint_environment_steps':step,
                                'completion_32_worlds':report['native_evaluation']['metrics']['chord_sr'],
                                'source_frames_covered':report['source_frames_covered'],'full_motion_test':False,
                                'training_reset_settling_steps':0,'training_object_support_force_limit_N':0.,'updated_utc':utc()}
                            print(json.dumps(state['checks'][key]),flush=True)
                        except Exception:
                            state['errors'][key]={'attempts':previous_error.get('attempts',0)+1,'updated_utc':utc(),'error':traceback.format_exc()}
                        state['updated_utc']=utc();save(result_path,state)
            state.update(updated_utc=utc(),phase='completed' if all_finished else 'watching_grasp_training')
            save(result_path,state)
            if all_finished:break
            time.sleep(15)
