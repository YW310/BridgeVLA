"""Detached, observation-only local depth evidence for action refinement.

Depth must be metric camera-Z depth, not radial range or a graphics Z-buffer.
Camera poses use the CV convention: X right, Y down, Z forward. No unseen
geometry is completed, and points behind an observed surface remain unknown.
"""

import math
import numbers

import torch

from ..geometry.torch_ops import (
    local_to_world_points, validate_rigid_transform, world_to_local_points,
)


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    # These parameters must remain usable by the float32 geometry calculations.
    converted = torch.tensor(value, dtype=torch.float32)
    if not torch.isfinite(converted).item() or converted.item() <= 0:
        raise ValueError(f'{name} must be positive and finite in float32')
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


@torch.no_grad()
def build_local_evidence(depth, intrinsics, camera_to_world, candidate_pose, *,
                         size_m=0.10, resolution=32, truncation_m=0.005,
                         chunk_size=8192):
    """Build a local projective-distance/evidence field from one depth frame.

    Args:
        depth: [B,V,H,W], camera-Z depth in meters. Nonfinite and nonpositive
            samples are missing observations, including a zero no-return value.
        intrinsics: [B,V,3,3], standard pinhole K, positive fx/fy. Skew is
            supported; the last row must be [0,0,1] and K[1,0] must be zero.
        camera_to_world: [B,V,4,4], rigid CV camera-to-world transforms.
        candidate_pose: [B,4,4], rigid local/gripper-to-world transforms.
        size_m: Side length of the cubic local field, in meters.
        resolution: Number of voxel centers per spatial axis.
        truncation_m: Projective camera-Z tolerance around each depth sample.
        chunk_size: Maximum number of voxels projected in a temporary chunk.

    All tensors must be on the same device. Calculation and outputs are detached
    float32. Spatial output axes D,H,W correspond to local Z,Y,X; the first and
    last voxel centers lie half a voxel inside the cube boundary.

    Each voxel is transformed to each camera and sampled at its nearest pixel.
    Only projections within pixel-center bounds [0,W-1] x [0,H-1], in front of
    the camera, with valid depth contribute. Let delta = depth - camera_z:
      * delta > truncation_m: free-space support;
      * abs(delta) <= truncation_m: near-surface support;
      * delta < -truncation_m: occluded/unknown, not occupied.

    Returns:
        dict with field [B,5,D,H,W] and known_fraction [B]. Channels are:
          0: mean clipped delta/truncation over free/near-surface supports;
             zero if no support. This is projective distance, not a true SDF.
          1: near-surface support count divided by the total number of views;
          2: free-space support count divided by the total number of views;
          3: known indicator, one if any free/near-surface support exists;
          4: conflict indicator, one if both support types exist across views.
    Conflict is a mixed-evidence diagnostic, not proof of a calibration error.
    Near-surface support describes a projective tolerance band, not certified
    occupancy. No interpolation across depth discontinuities is performed.
    """
    size_m = _positive_float(size_m, 'size_m')
    truncation_m = _positive_float(truncation_m, 'truncation_m')
    resolution = _positive_int(resolution, 'resolution')
    chunk_size = _positive_int(chunk_size, 'chunk_size')
    tensors = {'depth': depth, 'intrinsics': intrinsics,
               'camera_to_world': camera_to_world, 'candidate_pose': candidate_pose}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.is_complex():
            raise ValueError(f'{name} must be a real torch.Tensor')
    if depth.ndim != 4 or any(dimension <= 0 for dimension in depth.shape):
        raise ValueError('depth must have nonempty shape [B,V,H,W]')
    batch, views, height, width = depth.shape
    if intrinsics.shape != (batch, views, 3, 3):
        raise ValueError('intrinsics must have shape [B,V,3,3]')
    if camera_to_world.shape != (batch, views, 4, 4):
        raise ValueError('camera_to_world must have shape [B,V,4,4]')
    if candidate_pose.shape != (batch, 4, 4):
        raise ValueError('candidate_pose must have shape [B,4,4]')
    if any(tensor.device != depth.device for tensor in tensors.values()):
        raise ValueError('all geometry inputs must be on the same device')
    depth = depth.detach().float()
    intrinsics = intrinsics.detach().float()
    camera_to_world = camera_to_world.detach().float()
    candidate_pose = candidate_pose.detach().float()
    if not torch.isfinite(intrinsics).all().item():
        raise ValueError('intrinsics must be finite')
    expected_last = intrinsics.new_tensor([0., 0., 1.]).expand_as(intrinsics[..., 2, :])
    if (not torch.allclose(intrinsics[..., 2, :], expected_last, atol=1e-6, rtol=0)
            or not torch.allclose(intrinsics[..., 1, 0],
                                  torch.zeros_like(intrinsics[..., 1, 0]),
                                  atol=1e-6, rtol=0)
            or not (intrinsics[..., 0, 0] > 0).all().item()
            or not (intrinsics[..., 1, 1] > 0).all().item()):
        raise ValueError('intrinsics must be standard pinhole K with positive fx/fy')
    for name, pose in (('camera_to_world', camera_to_world), ('candidate_pose', candidate_pose)):
        validate_rigid_transform(pose, name, rotation_rtol=1e-4, homogeneous_atol=1e-5)

    coordinate = ((torch.arange(resolution, device=depth.device, dtype=torch.float32)
                   + 0.5) / resolution - 0.5) * size_m
    grid_z, grid_y, grid_x = torch.meshgrid(coordinate, coordinate, coordinate,
                                          indexing='ij')
    local_points = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)
    field = depth.new_zeros(batch, 5, local_points.shape[0])
    depth_flat = depth.flatten(-2)
    camera_rotation = camera_to_world[..., :3, :3]
    camera_translation = camera_to_world[..., :3, 3]
    local_rotation = candidate_pose[:, :3, :3]
    local_translation = candidate_pose[:, :3, 3]

    for start in range(0, local_points.shape[0], chunk_size):
        local = local_points[start:start + chunk_size]
        world = local_to_world_points(local[None].expand(batch, -1, -1),
                                      local_translation, local_rotation)
        # Each camera is an independent frame for the same world voxel centers.
        camera = world_to_local_points(
            world[:, None].expand(-1, views, -1, -1).reshape(batch * views, -1, 3),
            camera_translation.reshape(batch * views, 3),
            camera_rotation.reshape(batch * views, 3, 3),
        ).reshape(batch, views, -1, 3)
        camera_z = camera[..., 2]
        in_front = torch.isfinite(camera).all(-1) & (camera_z > 0)
        safe_z = torch.where(in_front, camera_z, torch.ones_like(camera_z))
        homogeneous = torch.einsum('bvij,bvnj->bvni', intrinsics, camera)
        pixel_xy = homogeneous[..., :2] / safe_z[..., None]
        inside = (in_front & torch.isfinite(pixel_xy).all(-1)
                  & (pixel_xy[..., 0] >= 0) & (pixel_xy[..., 0] <= width - 1)
                  & (pixel_xy[..., 1] >= 0) & (pixel_xy[..., 1] <= height - 1))
        safe_xy = torch.where(inside[..., None], pixel_xy, torch.zeros_like(pixel_xy))
        nearest = torch.floor(safe_xy + 0.5).long()
        index = nearest[..., 1] * width + nearest[..., 0]
        sampled = depth_flat.gather(-1, index)
        valid = inside & torch.isfinite(sampled) & (sampled > 0)
        delta = torch.where(valid, sampled - camera_z, torch.zeros_like(sampled))
        near = valid & (delta.abs() <= truncation_m)
        free = valid & (delta > truncation_m)
        observed = near | free
        support_count = observed.sum(1)
        near_count = near.sum(1)
        free_count = free.sum(1)
        normalized = (delta / truncation_m).clamp(-1, 1)
        distance = torch.where(observed, normalized, torch.zeros_like(normalized)).sum(1)
        distance = distance / support_count.clamp_min(1)
        end = start + local.shape[0]
        field[:, 0, start:end] = distance
        field[:, 1, start:end] = near_count.float() / views
        field[:, 2, start:end] = free_count.float() / views
        field[:, 3, start:end] = (support_count > 0).float()
        field[:, 4, start:end] = ((near_count > 0) & (free_count > 0)).float()

    known_fraction = field[:, 3].mean(-1)
    return {'field': field.reshape(batch, 5, resolution, resolution, resolution),
            'known_fraction': known_fraction}
