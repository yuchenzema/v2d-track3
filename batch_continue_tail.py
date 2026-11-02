"""Continue the secondary single-object queue after the current worker exits."""
from pathlib import Path
import os,subprocess,sys,time
base=Path('/home/yu_eli/v2d-track3');pid=int(sys.argv[1])
while True:
 try:
  os.kill(pid,0)
  if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
 except (ProcessLookupError,FileNotFoundError):break
 time.sleep(15)
subprocess.run([sys.executable,'-u',str(base/'batch_tail_training.py'),'35','36'],cwd=base,check=True)
