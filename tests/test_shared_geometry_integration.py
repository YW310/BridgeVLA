"""Cross-backend coordinate conventions used by OHT chunks and pose refinement."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from finetune.bridgevla.geometry.numpy_ops import backproject, rotation_error
from finetune.bridgevla.geometry.torch_ops import (
    local_to_world_points, quaternion_matrix, rotation_matrix_error_rad,
    world_to_local_points,
)
from finetune.bridgevla.models.local_geometry import build_local_evidence


def test_numpy_depth_points_and_torch_frames_agree_after_camera_rotation():
    intrinsics = np.array([[120., 3., 1.], [0., 130., 1.], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [35., -27., 81.], degrees=True).as_matrix()
    transform[:3, 3] = [.3, -.2, .9]
    depth = np.full((3, 3), .7)
    world = torch.from_numpy(backproject(depth, intrinsics, transform).reshape(1, -1, 3)).double()
    origin = torch.from_numpy(transform[:3, 3])[None]
    rotation = torch.from_numpy(transform[:3, :3])[None]
    optical = world_to_local_points(world, origin, rotation)
    np.testing.assert_allclose(optical.numpy()[0, :, 2], depth.ravel(), atol=1e-7)
    torch.testing.assert_close(local_to_world_points(optical, origin, rotation), world)
    # The center pixel is an observed surface after the exact same conversion.
    candidate = transform.copy()
    candidate[:3, 3] = world[0, 4].numpy()
    field = build_local_evidence(torch.from_numpy(depth)[None, None],
                                 torch.from_numpy(intrinsics)[None, None],
                                 torch.from_numpy(transform)[None, None],
                                 torch.from_numpy(candidate)[None], resolution=1,
                                 truncation_m=.001)['field']
    torch.testing.assert_close(field[0, 1:4, 0, 0, 0], torch.tensor([1., 0., 1.]))


def test_numpy_oht_and_torch_refinement_rotation_metrics_share_xyzw_and_radians():
    random = np.random.default_rng(38)
    first = Rotation.random(12, random_state=random).as_quat()
    second = Rotation.random(12, random_state=random).as_quat()
    first[:3] = [0., 0., 0., 1.]
    second[:3] = Rotation.from_rotvec([[0., 0., 0.], [np.pi, 0., 0.],
                                     [0., 1e-9, 0.]]).as_quat()
    expected = rotation_error(first, second)
    result = rotation_matrix_error_rad(quaternion_matrix(torch.from_numpy(first)),
                                       quaternion_matrix(torch.from_numpy(second)))
    np.testing.assert_allclose(result.numpy(), expected, atol=2e-12)
    signed = rotation_matrix_error_rad(quaternion_matrix(torch.from_numpy(-first)),
                                       quaternion_matrix(torch.from_numpy(second)))
    torch.testing.assert_close(result, signed, atol=0, rtol=0)


def test_spatial_regression_autocast_accepts_float32_measured_tcp_frames():
    from tests.test_oht_spatial_chunk import inputs

    decoder, context, _, _ = inputs('regression')
    decoder.eval()
    assert context[2].dtype == torch.float32
    with torch.autocast('cpu', dtype=torch.bfloat16):
        position, rotation, grip = decoder(*context)
    assert position.dtype == torch.float32
    assert rotation.dtype == torch.bfloat16
    for value in (position, rotation, grip):
        assert torch.isfinite(value).all()
    (position.square().sum() + rotation.float().square().sum() + grip.float().square().sum()).backward()
    assert decoder.head.weight.grad is not None
    assert torch.isfinite(decoder.head.weight.grad).all()
