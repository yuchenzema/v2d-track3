"""Continue the reviewed transition after the interrupted learner has finished cleanup."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
from batch_worker import BASE,PY,context
from batch_native import active_learners
from batch_generic_retarget import install
r,c=context(15);assert 3578518 not in active_learners();session=json.loads((r/'research/reference-training-session-transition.json').read_text());assert session['status']=='preparing';archive=r/session['archive'];checksums=json.loads((r/'coherent-input-checksums.json').read_text());decision=json.loads((r/'research/temporal-reference-selection.json').read_text());assert decision['accepted'];candidate=decision['candidate_reference'];selected_reference=session['new_reference'];folder=r/'flash_chord_coherent_actor_transfer'
for relative,digest in checksums.items():assert hashlib.sha256((r/relative).read_bytes()).hexdigest()==digest,relative
source=(BASE/'rebranch_episode15_training.py').read_text();continuation=source[source.index("assert 3511799 in active_learners()"):] ;exec(compile(continuation,str(BASE/'rebranch_episode15_training.py'), 'exec'))
