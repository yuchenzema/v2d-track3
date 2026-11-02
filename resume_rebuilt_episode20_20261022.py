"""Resume episode20's verified reconstruction and own saved training after reboot."""
import fcntl
import traceback
from rerun_four_episodes_20261022 import OUT, rebuild, train_and_export, status
from improvement_common import save, utc

if __name__ == '__main__':
    with (OUT / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rebuild(20)
            train_and_export(20)
        except Exception:
            error = traceback.format_exc()
            save(OUT / 'episode_000020-recovery-error.json', {'updated_utc': utc(), 'error': error})
            status(20, 'Episode20 recovery needs attention', state='failed', error=error)
            raise
