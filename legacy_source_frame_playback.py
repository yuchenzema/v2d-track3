"""Episode-local legacy HandPose playback adapter; no interpolation or pose edits.

motion_v1's generic robot interface cannot bind SharpaHands. Preserve the
validated legacy HandPose interface and use the same source-frame metadata
semantics as motion_v1. This adapter does not modify the official evaluator.
"""
from dataclasses import replace
import importlib
import ast


def prepare_script(source,config):
    """Attach the input adapter to generated entrypoints, preserving future imports."""
    if not config.get('validated_video_pipeline') or 'install_source_frame_playback()' in source:return source
    tree=ast.parse(source);line=0
    if tree.body and isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
        line=tree.body[0].end_lineno
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='__future__':line=max(line,node.end_lineno)
    lines=source.splitlines(keepends=True)
    lines.insert(line,'\nfrom legacy_source_frame_playback import install as install_source_frame_playback\ninstall_source_frame_playback()\n')
    return ''.join(lines)


def preserve_source_frames(load_legacy,path,control_fps=None,motion_speed=1.):
    if motion_speed!=1.:raise ValueError('Source-frame playback requires motion_speed=1.0')
    reference=load_legacy(path,control_fps=None,motion_speed=1.)
    return replace(reference,_metadata=replace(reference.metadata,is_resampled=False,source_frame_playback=True),
                   _fps=control_fps if control_fps is not None else reference.fps)


def install():
    data=importlib.import_module('flash_chord.data')
    loader=importlib.import_module('flash_chord.data.loader')
    mano=importlib.import_module('flash_chord.data.mano_sharpa')
    original=loader.load_reference
    if getattr(original,'_legacy_source_frame_adapter',False):return
    def load(parquet_path,control_fps=None,motion_speed=1.,source_frame_playback=False):
        try:
            return original(parquet_path,control_fps=control_fps,motion_speed=motion_speed,
                            source_frame_playback=source_frame_playback)
        except ValueError as error:
            if not source_frame_playback or str(error)!='source-frame playback is supported only for motion_v1 references':raise
            return preserve_source_frames(mano.load_mano_sharpa,parquet_path,control_fps,motion_speed)
    load._legacy_source_frame_adapter=True
    loader.load_reference=load;data.load_reference=load
    setup=importlib.import_module('flash_chord.scene.setup')
    setup.load_reference=load
