"""The policy tensor interface, without RLBench or PerAct imports."""

DEFAULT_CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
DEFAULT_SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]


def stack_on_channel(value):
    """[B,T,C,H,W] -> [B,T*C,H,W], matching the original replay layout."""
    if value.ndim != 5:
        raise ValueError("Camera tensors must have shape [B,T,C,H,W]")
    return value.reshape(value.shape[0], -1, *value.shape[-2:])


def preprocess_inputs(sample, cameras):
    observations, clouds = [], []
    for camera in cameras:
        rgb = stack_on_channel(sample[f"{camera}_rgb"]).float() / 127.5 - 1.0
        points = stack_on_channel(sample[f"{camera}_point_cloud"]).float()
        if rgb.shape != points.shape:
            raise ValueError(f"RGB/XYZ shape mismatch for {camera}")
        observations.append([rgb, points])
        clouds.append(points)
    return observations, clouds
