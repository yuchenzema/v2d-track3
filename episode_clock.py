"""Frame convention for new source-frame policies and retained legacy artifacts."""
def control_frames(config):
    return int(config['frames']) if config.get('validated_video_pipeline') else 2*(int(config['frames'])-1)


def motion_speed(config):
    return 1. if config.get('validated_video_pipeline') else .5
