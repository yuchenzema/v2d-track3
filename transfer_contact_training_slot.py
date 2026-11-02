"""Yield episode30's next saved checkpoint so episode10 can learn beside16."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc
from batch_native import _docker
OUT=BATCH/'improvements_20261020_contacts';record=OUT/'checkpoint-slot-transfer.json'
r=BASE/'runs/episode_000030_newton_grasp_curriculum_20261020';folder=r/'flash_chord_coherent_actor_transfer';step=30_000_128

def metadata(path):
 with path.open('rb') as f:
  length=int.from_bytes(f.read(8),'little');assert 0<length<16_000_000;header=json.loads(f.read(length))
 ends=[v['data_offsets'][1] for k,v in header.items() if k!='__metadata__'];assert path.stat().st_size==8+length+max(ends)
 envelope=header['__metadata__'];assert envelope.get('format')=='learner_state' and envelope.get('version')=='1'
 return {k.removeprefix('user.'):v for k,v in envelope.items() if k.startswith('user.')}
def update(**kw):
 old=json.loads(record.read_text());save(record,{**old,**kw,'updated_utc':utc()})
def top(container):return _docker(['top',container,'-eo','pid,args']).stdout

while True:
 files=[folder/f'{kind}_{step}.safetensors' for kind in ['policy','state']]
 if all(p.exists() and time.time()-p.stat().st_mtime>5 for p in files):break
 time.sleep(5)
meta=[metadata(p) for p in files];assert all(int(x['environment_steps'])==step for x in meta)
assert meta[0]['checkpoint_kind']=='policy' and meta[1]['checkpoint_kind']=='training_state'
assert meta[0]['checkpoint_set_id']==meta[1]['checkpoint_set_id']
priority=json.loads((BATCH/'training-priority.json').read_text());assert r.name not in priority['allowed_run_names']
assert '/episode/scripts/train_coherent_actor_transfer.py' in top('v2d-grasp-episode30-newton')
archive=r/f'research/checkpoint-preserved-for-contact-priority-{step}';archive.mkdir(exist_ok=True);hashes={}
for p in files:
 target=archive/p.name;shutil.copy2(p,target);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==hashes[p.name]
metrics=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1]);unsaved=max(0,metrics['environment_steps']-step)
code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Stopped only the episode30 learner at its preserved checkpoint:',found[0])
"""
print(_docker(['exec','v2d-grasp-episode30-newton','python','-c',code]).stdout,flush=True)
history=r/'research/training-resume-sessions.json';rows=json.loads(history.read_text()) if history.exists() else []
rows.append({'utc':utc(),'reason':'Yield GPU slot to the latest requested episode10; resume the same full-motion experiment afterwards.',
 'preserved_environment_steps':step,'latest_unsaved_metric_environment_steps':metrics['environment_steps'],'unsaved_environment_steps_to_repeat':unsaved,
 'checkpoint_archive':str(archive.relative_to(r)),'checkpoint_sha256':hashes,'optimizer_and_critic_preserved_in_checkpoint':True,
 'training_completed':False,'continuation_pending':True})
save(history,rows);update(state='waiting_for_episode10_initialization',preserved_checkpoint=str(archive),checkpoint_sha256=hashes,unsaved_environment_steps_to_repeat=unsaved)
while True:
 r10=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020';m=r10/'flash_chord_coherent_actor_transfer/metrics.jsonl';runtime=r10/'research/contact-finetune-zero-support-runtime.json'
 lines=m.read_text().splitlines() if m.exists() else []
 d=json.loads(lines[-1]) if lines else {};checked=json.loads(runtime.read_text()) if runtime.exists() else {}
 if d.get('additional_environment_steps',0)>0 and checked.get('world_count')==512 and '/episode/scripts/train_coherent_actor_transfer.py' in top('v2d-grasp-episode10-newton'):break
 time.sleep(10)
p=BATCH/'training-priority.json';priority=json.loads(p.read_text());priority['allowed_run_names']=list(dict.fromkeys(priority.get('allowed_run_names',[])+[r.name]));priority['temporarily_deferred_run_names']=[n for n in priority.get('temporarily_deferred_run_names',[]) if n!=r.name];priority['updated_utc']=utc();save(p,priority)
update(state='slot_transferred_episode30_queued_for_continuation',episode10_training_started=True,episode10_first_reviewed_runtime_world_count=512,
 episode10_metrics_environment_steps=d['environment_steps'],episode30_readmission_restored=True)
print('10 is learning alongside16;30 checkpoint is preserved and its continuation is queued.',flush=True)
