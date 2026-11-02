"""Run an unchanged evaluator with a declared source-reference input adapter."""
import runpy
import sys
from legacy_source_frame_playback import install

if __name__=='__main__':
    entrypoint=sys.argv[1]
    sys.argv=[entrypoint,*sys.argv[2:]]
    install()
    from official_body_recording import install as install_body_recording
    install_body_recording()
    runpy.run_path(entrypoint,run_name='__main__')
