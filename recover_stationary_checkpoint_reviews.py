"""Recover reviews after an initial checkpoint retained its previous reference pathname."""
from pathlib import Path
import threading,time,traceback
from batch_review import monitor
from improvement_common import BASE
R=BASE/'runs/episode_000016_newton_stationary_hold_20261020';stop=threading.Event()
def final_handoff():
 while not stop.is_set():
  if (R/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():stop.set();return
  p=R/'research/improvement-status.json'
  if p.exists():
   import json
   if json.loads(p.read_text()).get('status')=='failed':stop.set();return
  stop.wait(1)
threading.Thread(target=final_handoff,daemon=True).start()
try:monitor(16,root_override=R,container_override='v2d-stationary-episode16-newton',stop_event=stop)
except RuntimeError:
 if not stop.is_set():raise
 print('Checkpoint reviews handed back to the main continuation worker for finalization.',flush=True)
