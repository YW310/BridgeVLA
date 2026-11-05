"""Run real render Python/camera code with only the CUDA backend replaced."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / 'finetune/bridgevla/libs/point-renderer'


def renderer_scope(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from point_renderer import ops
    from point_renderer.cameras import OrthographicCameras, PerspectiveCameras
    calls = []
    def render(indices, depths, features, image, depth, default_depth, default_color):
        assert indices.numel() > 0, 'Empty input must bypass the CUDA launch'
        calls.append((indices.clone(), depths.clone()))
        image.fill_(default_color)
        depth.fill_(default_depth)
    scope = dict(torch=torch, math=math, ops=ops, OrthographicCameras=OrthographicCameras,
                 PerspectiveCameras=PerspectiveCameras,
                 r=SimpleNamespace(render_feature_pointcloud_to_image=render))
    for file, names in (('renderer.py', {'_prep_render_batch_inputs', 'PointRenderer'}),
                        ('rvt_renderer.py', {'RVTBoxRenderer'})):
        path = ROOT / 'point_renderer' / file
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
        for node in nodes:
            if isinstance(node, ast.FunctionDef):
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
