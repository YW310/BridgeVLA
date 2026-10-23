"""Label-free sensor snapshots and bounded CPU geometry for online diagnostics.

Raw snapshots retain the wire RGB and metric Z-depth before training resize,
rounding, workspace cropping, or point filtering. Only NumPy and Pillow are used;
this module neither imports the simulator nor runs a second policy forward.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


MAX_SCENE_POINTS = 24000
_OPTICAL_TO_OPENGL = np.diag([1., -1., -1., 1.])
_MARKERS = (('policy', '#ff4d4d'), ('heatmap', '#40e0ff'), ('measured', '#55e67e'))


def _finite(value, shape, name):
    value = np.asarray(value, np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite {shape}')
    return value


def _pose_matrix_xyzw(pose):
    pose = _finite(pose, (7,), 'pose')
    quaternion = pose[3:]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError('Zero quaternion')
    x, y, z, w = quaternion / norm
    transform = np.eye(4)
    transform[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    transform[:3, 3] = pose[:3]
    return transform


def _camera(rgb, depth, intrinsics, extrinsics, kind, limits):
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Sensor RGB must be uint8 HWC')
    if depth.shape != rgb.shape[:2] or min(depth.shape) < 1:
        raise ValueError('Sensor RGB and depth must be aligned and nonempty')
    matrix = _finite(intrinsics, (3, 3), 'intrinsics')
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError('Invalid camera intrinsics')
    # Skew is allowed, but a singular K is not.
    np.linalg.inv(matrix)
    transform = _finite(extrinsics, (4, 4), 'world_from_optical')
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1]) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError('Camera extrinsic must be a rigid optical-to-world transform')
    if kind not in ('z', 'ray'):
        raise ValueError('Sensor depth kind must be z or ray')
    limits = _finite(limits, (2,), 'depth_limits')
    if limits[0] <= 0 or limits[1] <= limits[0]:
        raise ValueError('Depth limits must be ordered positive metres')
    return dict(rgb=rgb.copy(), depth=np.array(depth, np.float32, copy=True),
                intrinsics=matrix.copy(), world_from_optical=transform.copy(),
                depth_kind=kind, depth_limits=limits.tolist())


def raw_sensor_snapshot(values, config, tcp_pose=None):
    """Copy only configured raw RGB-D/calibration; ignore all other input fields.

    The raw HTTP wire format always sends metric Z and camera xyz+wxyz poses
    with OpenGL axes, regardless of the recording's configured depth kind.
    """
    cameras = {}
    for name, calibration in config['cameras'].items():
        pose = _finite(values[f'{name}_pose'], (7,), 'raw OpenGL pose')
        optical = _pose_matrix_xyzw(np.r_[pose[:3], pose[4:], pose[3]]) @ _OPTICAL_TO_OPENGL
        depth = calibration.get('depth', config.get('depth', {}))
        cameras[name] = _camera(values[f'{name}_rgb'], values[f'{name}_depth'],
                                values[f'{name}_intrinsics'], optical, 'z',
                                depth.get('limits', [.001, 10.]))
    snapshot = dict(source='raw_sensor', cameras=cameras)
    if tcp_pose is not None:
        snapshot['tcp_pose_xyzw'] = _finite(tcp_pose, (7,), 'TCP pose').copy()
    return snapshot


def prepared_sensor_snapshot(observation, config=None):
    """Fallback to canonical, potentially resized RGB-D when raw inputs are absent.

    Pass the resolved data profile to preserve canonical ray-vs-Z semantics.
    Without a profile the historical prepared-observation default is Z-depth.
    Already canonical optical extrinsics are never axis-flipped again.
    """
    config = config or {}
    configured = config.get('cameras')
    names = list(configured) if configured else [
        key[:-4] for key in observation if key.endswith('_rgb')]
    cameras = {}
    for name in names:
        fields = [f'{name}_{suffix}' for suffix in
                  ('rgb', 'depth', 'camera_intrinsics', 'camera_extrinsics')]
        if not all(field in observation for field in fields):
            continue
        rgb = np.asarray(observation[fields[0]])
        depth = np.asarray(observation[fields[1]])
        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError('Prepared RGB must be CHW')
        if depth.ndim != 3 or depth.shape[0] != 1:
            raise ValueError('Prepared depth must be 1xHxW')
        calibration = config.get('cameras', {}).get(name, {})
        depth_config = calibration.get('depth', config.get('depth', {}))
        cameras[name] = _camera(rgb.transpose(1, 2, 0), depth[0],
                                observation[fields[2]], observation[fields[3]],
                                depth_config.get('kind', 'z'), depth_config.get('limits', [.001, 10.]))
    if not cameras:
        return None
    snapshot = dict(source='prepared_observation', cameras=cameras)
    if observation.get('tcp_pose') is not None:
        snapshot['tcp_pose_xyzw'] = _finite(observation['tcp_pose'], (7,), 'TCP pose').copy()
    return snapshot


def _marker(name, color, value):
    if value is None:
        return None
    value = np.asarray(value, np.float64)
    if value.shape not in ((3,), (7,)) or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected a finite xyz point or xyz+xyzw pose')
    if value.shape == (7,):
        _pose_matrix_xyzw(value)  # Reject a non-renderable orientation.
    return dict(name=name, color=color, position=value[:3].tolist(),
                pose_xyzw=value.tolist() if value.shape == (7,) else None)


def _project(position, camera):
    transform = camera['world_from_optical']
    optical = (np.asarray(position) - transform[:3, 3]) @ transform[:3, :3]
    z = float(optical[2])
    result = dict(pixel=None, optical_depth=z, status='behind', sample_depth_z=None)
    if z <= 1e-8:
        return result
    projected = camera['intrinsics'] @ optical
    pixel = projected[:2] / projected[2]
    result['pixel'] = pixel.tolist()
    height, width = camera['depth'].shape
    if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        result['status'] = 'outside'
        return result
    x, y = np.floor(pixel).astype(int)
    sample = float(camera['depth'][y, x])
    low, high = camera['depth_limits']
    if not np.isfinite(sample) or not low <= sample <= high:
        result['status'] = 'invalid_depth'
        return result
    if camera['depth_kind'] == 'ray':
        ray = np.linalg.solve(camera['intrinsics'], [x, y, 1.])
        sample /= float(np.linalg.norm(ray))
    result['sample_depth_z'] = sample
    result['status'] = 'occluded' if z > sample + max(.005, .01 * sample) else 'visible'
    return result


def _frustum(camera):
    height, width = camera['depth'].shape
    pixels = np.array([[0, 0, 1], [width - 1, 0, 1],
                       [width - 1, height - 1, 1], [0, height - 1, 1.]])
    rays = pixels @ np.linalg.inv(camera['intrinsics']).T
    corners = np.concatenate([rays * .03, rays * .15])
    transform = camera['world_from_optical']
    return (corners @ transform[:3, :3].T + transform[:3, 3]).tolist()


def _budgets(counts):
    """Water-fill equal camera quotas without wasting empty-camera allocations."""
    budgets = np.zeros(len(counts), dtype=int)
    remaining = MAX_SCENE_POINTS
    while remaining > 0:
        active = [index for index, count in enumerate(counts) if budgets[index] < count]
        if not active:
            break
        share = max(1, remaining // len(active))
        for index in active:
            take = min(share, counts[index] - budgets[index], remaining)
            budgets[index] += take
            remaining -= take
            if remaining == 0:
                break
    return budgets


def _scene_points(cameras):
    valid_indices = []
    for camera in cameras:
        depth = camera['depth']
        low, high = camera['depth_limits']
        valid_indices.append(np.flatnonzero(np.isfinite(depth) & (depth >= low) & (depth <= high)))
    budgets = _budgets([len(indices) for indices in valid_indices])
    points, colors, camera_ids = [], [], []
    for index, (camera, indices, budget) in enumerate(zip(cameras, valid_indices, budgets)):
        if budget == 0:
            continue
        # Sampling is deterministic and evenly spread over each camera's valid pixels.
        selected = indices[np.linspace(0, len(indices) - 1, int(budget), dtype=np.int64)]
        y, x = np.divmod(selected, camera['depth'].shape[1])
        rays = np.stack((x, y, np.ones_like(x)), axis=1) @ np.linalg.inv(camera['intrinsics']).T
        if camera['depth_kind'] == 'ray':
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        optical = rays * camera['depth'][y, x, None]
        transform = camera['world_from_optical']
        world = optical @ transform[:3, :3].T + transform[:3, 3]
        points.extend(np.round(world, 6).tolist())
        colors.extend(camera['rgb'][y, x].tolist())
        camera_ids.extend([index] * len(selected))
    return dict(positions=points, colors=colors, camera_indices=camera_ids)


def _depth_image(camera):
    depth = camera['depth']
    low, high = camera['depth_limits']
    valid = np.isfinite(depth) & (depth >= low) & (depth <= high)
    color = np.zeros((*depth.shape, 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        scaled = np.zeros_like(depth, np.float32)
        scaled[valid] = np.clip((depth[valid] - lo) / max(float(hi - lo), 1e-6), 0, 1)
        colors = np.stack((scaled, 1 - np.abs(2 * scaled - 1), 1 - scaled), axis=-1)
        color[valid] = np.round(colors[valid] * 255).astype(np.uint8)
    return Image.fromarray(color)


def _paint_markers(panel, projections, markers):
    draw = ImageDraw.Draw(panel)
    for marker in markers:
        projection = projections[marker['name']]
        if projection['status'] in ('behind', 'outside'):
            continue
        x, y = projection['pixel']
        radius = max(4, min(panel.size) // 80)
        color = marker['color']
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
        if projection['status'] != 'occluded':
            draw.line((x-radius-2, y, x+radius+2, y), fill=color, width=2)
            draw.line((x, y-radius-2, x, y+radius+2), fill=color, width=2)
        draw.text((x+radius+3, y-radius), marker['name'], fill=color,
                  stroke_width=1, stroke_fill='black')


def _montage(cameras, camera_records, markers, source):
    width = max(2 * camera['depth'].shape[1] for camera in cameras)
    header = 42
    height = header + sum(camera['depth'].shape[0] + 43 for camera in cameras)
    image = Image.new('RGB', (width, height), '#20252d')
    draw = ImageDraw.Draw(image)
    draw.text((6, 4), f'{source}: native sensor resolution; RGB | metric depth', fill='white')
    draw.text((6, 21), 'Red policy / cyan heatmap / green measured. Hollow: occluded. Depth: per-camera 2-98% scale.', fill='white')
    y = header
    for camera, record in zip(cameras, camera_records):
        label = record['name'].encode('ascii', errors='replace').decode('ascii')
        states = ', '.join(f'{name}={value["status"]}' for name, value in record['projections'].items())
        draw.text((6, y+3), f'{label}: {record["width"]}x{record["height"]} {record["depth_kind"]}', fill='white')
        draw.text((6, y+21), states, fill='#bec9d8')
        rgb, depth = Image.fromarray(camera['rgb']), _depth_image(camera)
        _paint_markers(rgb, record['projections'], markers)
        _paint_markers(depth, record['projections'], markers)
        image.paste(rgb, (0, y+43))
        image.paste(depth, (record['width'], y+43))
        y += record['height'] + 43
    return image


def export_sensor_scene(folder, entry_id, snapshot, policy_pose, heatmap_waypoint):
    """Write fixed-name sensor artifacts and bounded, calibrated browser geometry.

    Caller supplies a private generated entry directory and catches diagnostic
    failures independently of action inference. No camera name becomes a path.
    """
    if snapshot is None or not snapshot.get('cameras'):
        return None, None
    if snapshot.get('source') not in ('raw_sensor', 'prepared_observation'):
        raise ValueError('Unknown sensor snapshot source')
    folder = Path(folder)
    cameras, records, arrays = [], [], {}
    values = (policy_pose, heatmap_waypoint, snapshot.get('tcp_pose_xyzw'))
    markers = [marker for (name, color), value in zip(_MARKERS, values)
               if (marker := _marker(name, color, value)) is not None]
    for index, (name, source) in enumerate(snapshot['cameras'].items()):
        camera = _camera(source['rgb'], source['depth'], source['intrinsics'],
                         source['world_from_optical'], source['depth_kind'], source['depth_limits'])
        height, width = camera['depth'].shape
        projections = {marker['name']: _project(marker['position'], camera) for marker in markers}
        record = dict(index=index, name=str(name), width=width, height=height,
                      intrinsics=camera['intrinsics'].tolist(),
                      world_from_optical=camera['world_from_optical'].tolist(),
                      position=camera['world_from_optical'][:3, 3].tolist(),
                      depth_kind=camera['depth_kind'], depth_limits=camera['depth_limits'],
                      frustum=_frustum(camera), frustum_depths=[.03, .15], projections=projections)
        cameras.append(camera)
        records.append(record)
        for key in ('rgb', 'depth', 'intrinsics', 'world_from_optical'):
            arrays[f'camera_{index}_{key}'] = camera[key]
        for marker in markers:
            projection = projections[marker['name']]
            prefix = f'camera_{index}_{marker["name"]}'
            arrays[f'{prefix}_pixel'] = np.array(projection['pixel'] or [np.nan, np.nan], np.float64)
            arrays[f'{prefix}_status'] = np.array(projection['status'])
            arrays[f'{prefix}_optical_depth'] = np.array(projection['optical_depth'])
            arrays[f'{prefix}_sample_depth_z'] = np.array(
                projection['sample_depth_z'] if projection['sample_depth_z'] is not None else np.nan)
    arrays['camera_names'] = np.array([record['name'] for record in records])
    arrays['source'] = np.array(snapshot['source'])
    arrays['depth_kinds'] = np.array([camera['depth_kind'] for camera in cameras])
    for marker in markers:
        arrays[f'{marker["name"]}_world'] = np.array(marker['position'])
        if marker['pose_xyzw'] is not None:
            arrays[f'{marker["name"]}_pose_xyzw'] = np.array(marker['pose_xyzw'])
    scene = dict(version=1, source=snapshot['source'], coordinate_frame='env_origin_relative_world',
                 units='metres', points=_scene_points(cameras), cameras=records, markers=markers)
    (folder / 'scene.json').write_text(json.dumps(scene, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    np.savez_compressed(folder / 'sensors.npz', **arrays)
    _montage(cameras, records, markers, snapshot['source']).save(folder / 'sensors.png')
    base = f'/visualization/files/{entry_id}'
    return (dict(source=snapshot['source'], montage_url=f'{base}/sensors.png',
                 arrays_url=f'{base}/sensors.npz', cameras=records),
            dict(url=f'{base}/scene.json', point_count=len(scene['points']['positions']), source=snapshot['source']))
