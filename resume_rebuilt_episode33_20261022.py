"""Resume fresh33 after rigid source-supported spoon reconstruction repair."""
import traceback
from rerun_four_episodes_20261022 import rebuild,train_and_export,status,OUT
from improvement_common import save,utc

if __name__=='__main__':
 try:
  rebuild(33)
  train_and_export(33)
 except Exception:
  error=traceback.format_exc()
  save(OUT/'episode_000033-recovery-error.json',{'updated_utc':utc(),'error':error})
  status(33,'Fresh33 reconstruction/training recovery needs attention',state='failed',error=error)
  raise
