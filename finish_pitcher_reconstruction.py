"""Let the active pitcher pose estimator finish, then resume with updated workers."""
from pathlib import Path
import os,signal,subprocess,time,json
BASE=Path('/home/yu_eli/v2d-track3');parent=2385563;child=2416055
while Path(f'/proc/{child}/stat').exists():
 stat=Path(f'/proc/{child}/stat').read_text().split()
 if stat[2]=='Z':break
 time.sleep(10)
cmd=Path(f'/proc/{parent}/cmdline')
if cmd.exists() and b'batch_reconstruction_queue.py' in cmd.read_bytes():
 os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
subprocess.run([os.sys.executable,str(BASE/'batch_reconstruction_queue.py'),'20'],check=True)
