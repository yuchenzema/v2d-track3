"""Start reviewed training while the main queue finishes the preceding export.

The native runner owns the per-episode training lock and the global two-learner
admission gate. Checkpoint preservation and the main reviewer remain active.
"""
import json,subprocess,sys
from batch_worker import context,BASE
from batch_collision_reference import prepare
from batch_refine_contacts import refine
from batch_generic_retarget import install
from batch_reconstruction_queue import status
for ep in map(int,sys.argv[1:]):
 r,c=context(ep);assert c.get('reconstruction_visual_reviewed') and not c.get('reference_review_pending') and not c.get('terminal_support_repair_pending')
 if not (r/'coherent-input-checksums.json').exists():prepare(ep);refine(ep)
 install(r);status(r,'training','running',worker='reviewed preloader overlapping preceding export')
 subprocess.run([sys.executable,str(BASE/'batch_native.py'),str(ep),'train'],check=True)
