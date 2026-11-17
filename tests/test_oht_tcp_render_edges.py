"""Run real render Python/camera code with only the CUDA backend replaced."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / 'finetune/bridgevla/libs/point-renderer'


def renderer_scope(monkeypatch, *, scripted_projection=False):
    monkeypatch.syspath_prepend(str(ROOT))
    from point_renderer import ops
    from point_renderer.cameras import OrthographicCameras, PerspectiveCameras
    calls = []
    def render(indices, depths, features, image, depth, default_depth, default_color):
        assert indices.numel() > 0, 'Empty input must bypass the CUDA launch'
        assert indices.dtype == torch.int64
        assert depths.dtype == torch.float32, 'CUDA renderer requires float32 depths'
        assert features.dtype == torch.float32, 'CUDA renderer requires float32 features'
        assert image.dtype == depth.dtype == torch.float32
        calls.append((indices.clone(), depths.clone()))
        image.fill_(default_color)
        depth.fill_(default_depth)
    scope = dict(__name__=__name__, torch=torch, math=math, ops=ops, OrthographicCameras=OrthographicCameras,
                 PerspectiveCameras=PerspectiveCameras,
                 r=SimpleNamespace(render_feature_pointcloud_to_image=render))
    for file, names in (('renderer.py', {'_prep_render_batch_inputs', 'PointRenderer'}),
                        ('rvt_renderer.py', {'RVTBoxRenderer'})):
        path = ROOT / 'point_renderer' / file
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
        for node in nodes:
            if isinstance(node, ast.FunctionDef) and not scripted_projection:
                node.decorator_list = []  # Execute the same projection body without JIT/CUDA imports.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    point = scope['PointRenderer'].__new__(scope['PointRenderer'])
    point.device = 'cpu'
    point.timer = SimpleNamespace(reset=lambda: None, check=lambda *args: None)
    cameras = OrthographicCameras.from_lookat([[0., .5, 0.]], [[0., 0., 0.]], [[0., 0., 1.]],
                                             img_sizes_w=[2, 2], img_size_px=(8, 8))
    box = scope['RVTBoxRenderer'].__new__(scope['RVTBoxRenderer'])
    box.renderer, box.cameras, box.img_size = point, cameras, (8, 8)
    box.strict_input_device = False
    box.splat_radius, box.default_color, box.default_depth = None, 0., -1.
    box.aa_factor, box.normalize_output, box.with_depth = 1, True, True
    return point, box, calls, ops


@pytest.mark.parametrize('aa', [1, 2])
def test_empty_crop_bypasses_cuda_and_returns_finite_invalid_background(monkeypatch, aa):
    point, box, calls, _ = renderer_scope(monkeypatch)
    original = box.cameras.intrinsics.clone()
    image, depth = point.render_batch(torch.empty(0, 3), torch.empty(0, 10), box.cameras,
                                      (8, 8), default_depth=-1., aa_factor=aa)
    assert image.shape == (1, 8, 8, 10) and depth.shape == (1, 8, 8)
    assert (image == 0).all() and (depth == -1).all() and not calls
    torch.testing.assert_close(box.cameras.intrinsics, original)
    result = box(torch.empty(0, 3), torch.empty(0, 10), positive_depth_only=True)
    assert torch.isfinite(result).all() and (result[..., -1] == -1).all()
    assert not calls


def test_positive_depth_filter_and_tcp_camera_shift_leave_legacy_xy_unchanged(monkeypatch):
    point, box, calls, ops = renderer_scope(monkeypatch)
    points = torch.tensor([[0., .75, 0.], [0., -.75, 0.], [0., .5, 0.]])
    features = torch.ones(3, 3)
    original_pose = box.cameras.poses.clone()
    point.render_batch(points, features, box.cameras, (8, 8), positive_depth_only=True)
    indices, depths = calls[-1]
    assert indices[0] == -1 and indices[2] == -1 and indices[1] >= 0
    assert (depths >= 0).all()
    box(points, features)  # Legacy render does not opt into new clipping/camera placement.
    legacy_indices, legacy_depths = calls[-1]
    assert (legacy_indices >= 0).all() and legacy_depths[0] < 0
    box(points, features, positive_depth_only=True)
    tcp_indices, tcp_depths = calls[-1]
    torch.testing.assert_close(tcp_indices, legacy_indices)
    torch.testing.assert_close(tcp_depths, legacy_depths + 2)
    assert (tcp_depths > 0).all()
    torch.testing.assert_close(box.cameras.poses, original_pose)


@pytest.mark.parametrize('scripted_projection', [False, True])
@pytest.mark.parametrize('perspective', [False, True])
def test_bf16_autocast_preserves_float32_projection_and_outer_context(
        monkeypatch, scripted_projection, perspective):
    point, box, calls, _ = renderer_scope(monkeypatch, scripted_projection=scripted_projection)
    from point_renderer.cameras import PerspectiveCameras
    cameras = box.cameras
    if perspective:
        cameras = PerspectiveCameras.from_lookat(
            [[0., .5, 0.]], [[0., 0., 0.]], [[0., 0., 1.]], 60, (8, 8))
    points = torch.tensor([[.017, -.321, .051], [-.113, -.125, -.077]])
    features = torch.tensor([[.2, .4, .6], [.1, .3, .5]])
    baseline = point.render_batch(points, features, cameras, (8, 8))
    expected_indices, expected_depths = calls[-1]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        # The real renderer's unprotected einsum produces BF16 even with FP32
        # inputs. Reproduce that boundary rather than mocking its projection.
        projected = torch.einsum('ij,nj->ni', torch.eye(3), points)
        assert projected.dtype == torch.bfloat16
        actual = point.render_batch(points, features, cameras, (8, 8))
        assert torch.einsum('ij,nj->ni', torch.eye(3), points).dtype == torch.bfloat16
    actual_indices, actual_depths = calls[-1]
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_depths, expected_depths, rtol=0, atol=0)
    for result, expected in zip(actual, baseline):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('empty', [False, True])
def test_low_precision_render_inputs_are_normalized_without_mutating_callers(monkeypatch, dtype, empty):
    point, box, calls, _ = renderer_scope(monkeypatch, scripted_projection=True)
    points = torch.tensor([[.125, -.25, .0625]], dtype=dtype)[:0 if empty else 1]
    features = torch.tensor([[.25, .5, .75]], dtype=dtype)[:0 if empty else 1]
    # Cover externally supplied low-precision camera tensors as well. Do not
    # invert a BF16 camera: the renderer consumes the provided inverse matrix.
    box.cameras.inv_poses = box.cameras.inv_poses.to(dtype)
    box.cameras.intrinsics = box.cameras.intrinsics.to(dtype)
    poses_before, intrinsics_before = box.cameras.inv_poses.clone(), box.cameras.intrinsics.clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        image, depth = point.render_batch(points, features, box.cameras, (8, 8))
    assert image.dtype == depth.dtype == torch.float32
    assert points.dtype == features.dtype == dtype
    assert box.cameras.inv_poses.dtype == box.cameras.intrinsics.dtype == dtype
    torch.testing.assert_close(box.cameras.inv_poses, poses_before, rtol=0, atol=0)
    torch.testing.assert_close(box.cameras.intrinsics, intrinsics_before, rtol=0, atol=0)
    assert len(calls) == (0 if empty else 1)
