"""Reserve GPU capacity for the active fresh19 learner and its evaluations.
Only our reconstruction scheduler is held; any already running child finishes normally.
"""
import os,signal,time,json
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc
OUT=BATCH/'improvements_20261022_rebuild';parent=json.loads((OUT/'worker.json').read_text())['pid'];held=False

def alive(pid,script):
 try:return any(Path(os.fsdecode(x)).name==script for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x)
 except OSError:return False
try:
 while alive(parent,'rerun_four_episodes_20261022.py'):
  recovery=json.loads((OUT/'episode_000019-recovery-worker.json').read_text());busy=alive(recovery['pid'],recovery['script']);phase=json.loads((OUT/'episode_000019-status.json').read_text());waiting=busy and phase['status']!='failed'
  if waiting and not held:os.kill(parent,signal.SIGSTOP);held=True
  if not waiting and held:os.kill(parent,signal.SIGCONT);held=False
  save(OUT/'gpu-scheduling.json',{'updated_utc':utc(),'scheduler_pid':os.getpid(),'reconstruction_scheduler_pid':parent,'reconstruction_scheduler_waiting_for_gpu':held,'running_new_policy_episode':19 if waiting else None,'reason':'Fresh policy training and full-episode validation reserve GPU capacity; queued cold reconstruction resumes automatically. Existing children are allowed to finish normally.'})
  if not waiting:break
  time.sleep(10)
finally:
 if held and alive(parent,'rerun_four_episodes_20261022.py'):os.kill(parent,signal.SIGCONT)
 save(OUT/'gpu-scheduling.json',{'updated_utc':utc(),'scheduler_pid':os.getpid(),'reconstruction_scheduler_pid':parent,'reconstruction_scheduler_waiting_for_gpu':False,'independent_episode19_stage_finished':True})
