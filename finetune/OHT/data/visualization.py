"""CPU-only RGB-D/world geometry previews, separate from policy observations."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import CAMERAS, inside
from .point_filter import point_cloud_mask, point_filter_options
from .cloud_source import camera_cloud
from .object_cloud import box_mask, resolve_object_boxes


TARGET = np.array([35, 220, 75], dtype=np.uint8)
REFERENCE = np.array([65, 125, 255], dtype=np.uint8)
OVERLAP = np.array([255, 220, 50], dtype=np.uint8)
CURRENT = (0, 220, 255)
ACTION = (255, 70, 220)
CHUNK = (255, 160, 50)
PANEL_W, PANEL_H = 256, 192
MAX_PREVIEW_POINTS = 200_000
LOCAL_HALF_EXTENT = 0.20  # metres; diagnostic cube, not the model's stage-two crop
CAMERA_COLORS = dict(zip(CAMERAS, ((255, 85, 85), (85, 230, 130), (80, 150, 255),
                                  (255, 200, 65), (190, 100, 255))))
OBJECT_COLORS = ((255, 145, 40), (80, 180, 255), (190, 240, 60), (195, 125, 250),
                 (255, 105, 110), (80, 220, 170), (245, 220, 130), (180, 180, 255))
OBJECT_OVERLAP = (255, 255, 255)


def current_position_label(config):
    """The cached TCP is the recorded EE position only with an identity offset."""
    offset = config.get("link_to_tcp")
    return "EE-pos" if offset is not None and np.array_equal(offset, np.eye(4)) else "TCP"


def _object_tints(points, colors, boxes):
    colors = np.array(colors, dtype=np.uint8, copy=True)
    membership = np.zeros(len(points), dtype=np.int32)
    for box in boxes:
        keep = box_mask(points, box["bounds"])
        colors[keep] = box["color"]
        membership[keep] += 1
    colors[membership > 1] = OBJECT_OVERLAP
    return colors, membership > 0


def _object_cloud_overlay(observation, config, sample, specification, local_bounds):
    boxes, source = resolve_object_boxes(specification, sample, sample.get("diagnostic_contract_sha256"),
                                          allow_empty=True)
    chunks = []
    for camera in config["cameras"]:
        points, _ = camera_cloud(observation, config, camera, source)
        chunks.append(points[np.isfinite(points).all(axis=1)])
    points = np.concatenate(chunks)
    world = box_mask(points, config["scene_bounds"])
    local = np.zeros(len(points), dtype=bool) if local_bounds is None else box_mask(points, local_bounds)
    for box in boxes:
        keep = box_mask(points, box["bounds"])
        box.update(color=OBJECT_COLORS[box["index"] % len(OBJECT_COLORS)], source_points=int(keep.sum()),
                   world_points=int((keep & world).sum()), local_points=int((keep & local).sum()))
    colors, keep = _object_tints(points, np.zeros((len(points), 3), dtype=np.uint8), boxes)
    return dict(points=points[keep], colors=colors[keep], boxes=boxes, source=source)


def _object_view(overlay, bounds):
    if overlay is None:
        return None
    keep = box_mask(overlay["points"], bounds)
    points, colors = _sample_preview_cloud(overlay["points"][keep], overlay["colors"][keep])
    return dict(overlay, points=points, colors=colors)


def _object_label(box):
    label = f"object_{box['index']}"
    if box["name"] != label:
        label += " " + " ".join(box["name"].split())
    return label if len(label) <= 28 else label[:25]+"..."


def role_overlay(rgb, masks):
    """Blend only annotated regions: 30% RGB + 70% role color."""
    result = np.asarray(rgb, dtype=np.uint8).copy()
    empty = np.zeros(result.shape[:2], dtype=bool)
    target = np.asarray(masks.get("target", empty), dtype=bool)
    reference = np.asarray(masks.get("reference", empty), dtype=bool)
    if target.shape != empty.shape or reference.shape != empty.shape:
        raise ValueError("Visualization masks must match cached RGB resolution")
    for mask, color in ((target & ~reference, TARGET), (reference & ~target, REFERENCE),
                        (target & reference, OVERLAP)):
        result[mask] = (.3 * result[mask] + .7 * color).astype(np.uint8)
    return result


def depth_colors(depth, limits):
    """Color measured metric depth; invalid pixels stay black."""
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth >= limits[0]) & (depth <= limits[1])
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return rgb, None
    near, far = float(depth[valid].min()), float(depth[valid].max())
    values = (depth[valid] - near) / max(far - near, 1e-6)
    rgb[valid] = np.stack((values, 1 - np.abs(2 * values - 1), 1 - values), axis=-1) * 255
    return rgb, (near, far)


def project_world(points, intrinsics, world_from_optical, image_shape):
    """Project world points using cached optical extrinsics and scaled K."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    transform = np.asarray(world_from_optical)
    # Row-vector form of R.T @ (P_world - t); do not transpose R again here.
    optical = (points - transform[:3, 3]) @ transform[:3, :3]
    pixels = optical @ np.asarray(intrinsics).T
    xy = np.full((len(points), 2), np.nan)
    front = np.isfinite(optical).all(axis=1) & (optical[:, 2] > 1e-8)
    xy[front] = pixels[front, :2] / pixels[front, 2:3]
    height, width = image_shape
    visible = front & (xy[:, 0] >= 0) & (xy[:, 0] < width)
    visible &= (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return xy, visible


def _mark(draw, xy, color, label=None):
    x, y = map(float, xy)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
    draw.line((x - 8, y, x + 8, y), fill=color, width=2)
    draw.line((x, y - 8, x, y + 8), fill=color, width=2)
    if label:
        draw.text((x + 9, y - 16 if label in ("TCP", "EE-pos") else y + 6), label, fill=color)


def projection_status(pose, intrinsics, world_from_optical, image_shape):
    """Explain a missing marker; frustum membership is not depth visibility."""
    if pose is None:
        return "unavailable"
    point = np.asarray(pose, dtype=float)[:3]
    transform = np.asarray(world_from_optical)
    optical = (point - transform[:3, 3]) @ transform[:3, :3]
    if not np.isfinite(optical).all():
        return "invalid coordinates"
    if optical[2] <= 1e-8:
        return f"behind camera z={optical[2]:.3f}m"
    xy, inside_view = project_world(point, intrinsics, world_from_optical, image_shape)
    state = "in view" if inside_view[0] else "outside image"
    return f"{state} uv=({xy[0, 0]:.1f},{xy[0, 1]:.1f}) z={optical[2]:.3f}m"


def _chunk_data(targets, current_tcp):
    if targets is None:
        return None
    actions, valid = np.asarray(targets['action_chunk']), np.asarray(targets['action_valid'])
    if (actions.ndim != 2 or actions.shape[1] != 8 or valid.shape != actions.shape[:1]
            or valid.dtype != np.bool_ or not valid.any()
            or not np.array_equal(valid, np.arange(len(valid)) < valid.sum())):
        raise ValueError('Preview chunk requires [K,8] actions and a nonempty bool valid prefix')
    actions = actions[valid].copy()
    if not np.isfinite(actions).all():
        raise ValueError('Preview chunk has nonfinite valid actions')
    positions = actions[:, :3]
    if current_tcp is not None:
        positions = np.concatenate((np.asarray(current_tcp, dtype=float)[None, :3], positions))
    return dict(actions=actions, positions=positions, horizon=len(valid))


def _draw_chunk(draw, pixels, visible):
    # Keep original sequence indices: never connect across a point behind the
    # camera or outside the view/crop. Projection does not test occlusion.
    for index in range(1, len(pixels)):
        if visible[index - 1] and visible[index]:
            draw.line([tuple(pixels[index - 1]), tuple(pixels[index])], fill=CHUNK, width=2)
    for index in np.flatnonzero(visible):
        x, y = pixels[index]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=CHUNK)
        if index == len(pixels) - 1:
            draw.text((x + 4, y + 3), 'chunk end', fill=CHUNK)


def _camera_rgb(observation, camera, current_tcp, action_tcp, masks, specs, role_points, role_valid,
                current_label="TCP", chunk=None):
    raw = observation[f"{camera}_rgb"].transpose(1, 2, 0)
    image = Image.fromarray(role_overlay(raw, masks)).resize((PANEL_W, PANEL_H), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    height, width = raw.shape[:2]
    scale = np.array([PANEL_W / width, PANEL_H / height])
    K, transform = observation[f"{camera}_camera_intrinsics"], observation[f"{camera}_camera_extrinsics"]
    if chunk is not None:
        xy, visible = project_world(chunk['positions'], K, transform, (height, width))
        _draw_chunk(draw, xy * scale, visible)
    # Site regions are projected point hints, never painted as visible masks.
    if specs is not None:
        for index, (role, color) in enumerate((("target", TARGET), ("reference", REFERENCE))):
            if specs[role]["source"] == "site_region" and role_valid[index]:
                xy, visible = project_world(role_points[index], K, transform, (height, width))
                for point in xy[visible][::max(1, len(xy) // 128)]:
                    x, y = point * scale
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=tuple(map(int, color)))
    for pose, color, label in ((current_tcp, CURRENT, current_label), (action_tcp, ACTION, "goal")):
        if pose is not None:
            xy, visible = project_world(np.asarray(pose)[:3], K, transform, (height, width))
            if visible[0]:
                _mark(draw, xy[0] * scale, color, label)
    return image


def _orthographic(points, colors, bounds, axes, current_tcp, action_tcp, role_points, role_valid,
                  region_bounds=None, object_overlay=None, current_label="TCP", chunk=None):
    width, height = 340, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    bounds = np.asarray(bounds)
    lower, span = bounds[:3], bounds[3:] - bounds[:3]
    horizontal, vertical = axes

    def pixels(cloud):
        uv = (np.asarray(cloud) - lower) / span
        return np.stack((uv[:, horizontal] * (width - 1),
                         (1 - uv[:, vertical]) * (height - 1)), axis=-1)

    points, colors = np.asarray(points), np.asarray(colors)
    if object_overlay is not None:
        colors, _ = _object_tints(points, colors, object_overlay["boxes"])
        # Combine both sources in one depth test. Box points behind a foreground
        # surface must stay hidden; same-surface points retain the object tint.
        points = np.concatenate((object_overlay["points"], points))
        colors = np.concatenate((object_overlay["colors"], colors))
    valid = np.isfinite(points).all(axis=1) & (points >= bounds[:3]).all(axis=1)
    valid &= (points < bounds[3:]).all(axis=1)
    points, colors = points[valid], colors[valid]
    if len(points):
        xy = pixels(points).astype(int)
        normal = next(axis for axis in range(3) if axis not in axes)
        # One display pixel per projected point, without expanding neighbours.
        # Resolve collisions by depth, viewed from the positive normal axis.
        order = np.argsort(points[:, normal], kind="stable")[::-1]
        flat = xy[:, 1] * width + xy[:, 0]
        _, first = np.unique(flat[order], return_index=True)
        chosen = order[first]
        image[xy[chosen, 1], xy[chosen, 0]] = colors[chosen]
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    if region_bounds is not None:
        intersection = np.r_[np.maximum(bounds[:3], region_bounds[:3]),
                             np.minimum(bounds[3:], region_bounds[3:])]
        if (intersection[3:] > intersection[:3]).all():
            corners = pixels(np.array([intersection[:3], intersection[3:]]))
            draw.rectangle((*corners.min(axis=0), *corners.max(axis=0)), outline=ACTION, width=1)
    if object_overlay is not None:
        for box in object_overlay["boxes"]:
            extent = np.asarray(box["bounds"])
            intersection = np.r_[np.maximum(bounds[:3], extent[:3]), np.minimum(bounds[3:], extent[3:])]
            if (intersection[3:] > intersection[:3]).all():
                corners = pixels(np.array([intersection[:3], intersection[3:]]))
                draw.rectangle((*corners.min(axis=0), *corners.max(axis=0)), outline=box["color"], width=1)
            center = np.asarray(box["center"])
            if (center >= bounds[:3]).all() and (center < bounds[3:]).all():
                label = _object_label(box) + (" (empty)" if not box["source_points"] else "")
                xy = pixels(center[None])[0]
                _mark(draw, xy, box["color"])
                text_bounds = draw.textbbox((0, 0), label)
                # Leave goal/TCP labels around the cross and keep names fully
                # inside the panel when an object center approaches an edge.
                position = (max(4, min(xy[0]+9, width-text_bounds[2]-4)),
                            max(36, min(xy[1]+18, height-text_bounds[3]-4)))
                draw.text(position, label, fill=box["color"])
    if role_points is not None:
        for index, color in enumerate((TARGET, REFERENCE)):
            if role_valid[index]:
                cloud = role_points[index]
                valid = np.isfinite(cloud).all(axis=1) & (cloud >= bounds[:3]).all(axis=1)
                valid &= (cloud < bounds[3:]).all(axis=1)
                for x, y in pixels(cloud[valid]):
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=tuple(map(int, color)))
    if chunk is not None:
        path = chunk['positions']
        visible = np.isfinite(path).all(axis=1) & (path >= bounds[:3]).all(axis=1) & (path < bounds[3:]).all(axis=1)
        _draw_chunk(draw, pixels(path), visible)
    for pose, color, label in ((current_tcp, CURRENT, current_label), (action_tcp, ACTION, "goal")):
        if pose is not None:
            point = np.asarray(pose)[:3]
            if np.isfinite(point).all() and (point >= bounds[:3]).all() and (point < bounds[3:]).all():
                _mark(draw, pixels(point[None])[0], color, label)
    names = "XYZ"
    draw.text((4, 4), f"{names[horizontal]}: {bounds[horizontal]:.3f} .. {bounds[horizontal+3]:.3f} m", fill="white")
    draw.text((4, 20), f"{names[vertical]}: {bounds[vertical]:.3f} .. {bounds[vertical+3]:.3f} m (up)", fill="white")
    return result


def _sample_preview_cloud(points, colors):
    if len(points) > MAX_PREVIEW_POINTS:
        indices = np.linspace(0, len(points)-1, MAX_PREVIEW_POINTS).astype(int)
        return points[indices], colors[indices]
    return points, colors


def save_preview(path, observation, config, sample, current_tcp=None, role_specs=None,
                 role_masks=None, role_points=None, role_valid=None, color_by_camera=False, object_boxes=None,
                 chunk_targets=None):
    """Save sampled diagnostics without changing cached tensors or labels."""
    current_label = current_position_label(config)
    chunk = _chunk_data(chunk_targets, current_tcp)
    cameras = list(config["cameras"])
    goal_pose = sample.get("labels", {}).get("gripper_pose")
    action_tcp = None if goal_pose is None else np.asarray(goal_pose)
    bounds = np.asarray(config["scene_bounds"])
    filter_options = point_filter_options(config.get("point_cloud_filter"))
    chunks, rgb_chunks = [], []
    for camera in cameras:
        cloud = observation[f"{camera}_point_cloud"].reshape(3, -1).T
        colors = observation[f"{camera}_rgb"].reshape(3, -1).T
        if color_by_camera:
            colors = np.broadcast_to(CAMERA_COLORS[camera], colors.shape).astype(np.uint8)
        valid = point_cloud_mask(cloud, filter_options) & (cloud >= bounds[:3]).all(axis=1)
        valid &= (cloud < bounds[3:]).all(axis=1)
        chunks.append(cloud[valid])
        rgb_chunks.append(colors[valid])
    points, colors = np.concatenate(chunks), np.concatenate(rgb_chunks)
    # Select the local ROI before global display sampling so small objects do
    # not disappear merely because distant scene points exhausted the budget.
    local_bounds = None if action_tcp is None else np.r_[action_tcp[:3] - LOCAL_HALF_EXTENT, action_tcp[:3] + LOCAL_HALF_EXTENT]
    object_overlay = None if object_boxes is None else _object_cloud_overlay(observation, config, sample,
                                                                            object_boxes, local_bounds)
    local_mask = np.zeros(len(points), dtype=bool) if local_bounds is None else (
        (points >= local_bounds[:3]).all(axis=1) & (points < local_bounds[3:]).all(axis=1))
    local_count, global_count = int(local_mask.sum()), len(points)
    local_points, local_colors = _sample_preview_cloud(points[local_mask], colors[local_mask])
    points, colors = _sample_preview_cloud(points, colors)
    row_height, top = PANEL_H + 44, 96
    ortho_top = top + ((len(cameras) + 1) // 2) * row_height
    footer_top = ortho_top + (638 if action_tcp is not None else 318)
    chunk_top = footer_top
    if chunk is not None:
        footer_top += 344
    object_top = footer_top
    if object_overlay is not None:
        footer_top += 24 + max(1, len(object_overlay["boxes"])) * 318
    canvas = Image.new("RGB", (1024, footer_top + 48), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    frame = "unknown" if sample.get("frame") is None else sample["frame"]
    target_frame = "unknown" if sample.get("target_frame") is None else sample["target_frame"]
    timestamp = "unknown" if sample.get("timestamp") is None else f"{sample['timestamp']:.3f} s"
    draw.text((8, 6), f"OHT {sample['id']} | {sample.get('split', 'unknown')} | frame {frame} -> {target_frame}", fill="white")
    draw.text((8, 24), f"time {timestamp} | {sample.get('goal', 'instruction unavailable')}", fill="white")
    current = "not cached" if current_tcp is None else np.array2string(np.asarray(current_tcp)[:3], precision=3)
    goal = "not cached" if action_tcp is None else np.array2string(action_tcp[:3], precision=3)
    draw.text((8, 42), f"World {current_label} {current} -> GT goal {goal} m", fill="white")
    legend = (f"Cyan=current {current_label}; magenta=future action; orthographic colors=physical camera (NOT roles)"
              if color_by_camera else
              f"Cyan=current {current_label}; magenta=future action; green=Target; blue=Reference; yellow=overlap")
    if chunk is not None:
        legend = f"Cyan=current {current_label}; magenta=next key goal; orange=GT future chunk (not collision/visibility checked)"
    draw.text((8, 60), legend, fill="white")
    if role_specs is not None:
        status = " | ".join(f"{role}: {spec['source']} present={spec['present']} known={spec['known']} geometry={bool(role_valid[i])}"
                            for i, (role, spec) in enumerate((('target', role_specs['target']), ('reference', role_specs['reference']))))
        draw.text((8, 78), status, fill="white")
    elif sample.get("diagnostic_note"):
        draw.text((8, 78), f"Diagnostic only: replay completeness not checked; direct NPZ may have no {current_label}/goal metadata.", fill=ACTION)
    elif chunk is not None:
        draw.text((8, 78), f"GT dense chunk: {len(chunk['actions'])}/{chunk['horizon']} valid future frames; padding omitted; zoomed path below", fill=CHUNK)
    for index, camera in enumerate(cameras):
        x, y = (index % 2) * 512, top + (index // 2) * row_height
        masks = {role: values[camera] for role, values in (role_masks or {}).items()}
        draw.text((x + 4, y + 2), camera + (" RGB + roles" if role_specs is not None else " RGB"), fill="white")
        canvas.paste(_camera_rgb(observation, camera, current_tcp, action_tcp, masks,
                                role_specs, role_points, role_valid, current_label=current_label, chunk=chunk), (x, y + 20))
        K = observation[f"{camera}_camera_intrinsics"]
        transform = observation[f"{camera}_camera_extrinsics"]
        image_shape = observation[f"{camera}_rgb"].shape[1:]
        for offset, pose, color, label in ((214, current_tcp, CURRENT, current_label),
                                          (225, action_tcp, ACTION, "goal")):
            draw.text((x + 4, y + offset), label + ": " + projection_status(pose, K, transform, image_shape), fill=color)
        depth, limits = depth_colors(observation[f"{camera}_depth"][0], config["depth"].get("limits", [0.001, 10]))
        draw.text((x + PANEL_W + 4, y + 2), camera + " metric depth", fill="white")
        canvas.paste(Image.fromarray(depth).resize((PANEL_W, PANEL_H), Image.Resampling.NEAREST), (x + PANEL_W, y + 20))
        if limits is None:
            text = "No valid depth"
        elif limits[1] - limits[0] < 1e-6:
            text = f"Constant depth {limits[0]:.3f} m"
        else:
            text = f"Blue near {limits[0]:.3f} m / red far {limits[1]:.3f} m"
        draw.text((x + PANEL_W + 4, y + 214), text, fill="white")
        if chunk is not None:
            _, visible = project_world(chunk['actions'][:, :3], K, transform, image_shape)
            draw.text((x + PANEL_W + 4, y + 225), f"GT chunk: {int(visible.sum())}/{len(visible)} points in image", fill=CHUNK)
    filter_status = " | manual XYZ filter ON" if filter_options["enabled"] else ""
    draw.text((8, ortho_top + 2),
              f"Global workspace | in-bounds points {global_count} -> shown {len(points)} | 1px points" + filter_status, fill="white")
    local_top = ortho_top + 318
    views = [(ortho_top, points, colors, bounds, local_bounds)]
    if action_tcp is not None:
        draw.text((8, local_top + 2),
                  f"GT-centered refine diagnostic (NOT model stage2) | +/-{LOCAL_HALF_EXTENT:.2f} m | points {local_count} -> {len(local_points)}",
                  fill="white")
        views.append((local_top, local_points, local_colors, local_bounds, None))
    for row_top, cloud, rgb, view_bounds, roi in views:
        for index, axes in enumerate(((0, 1), (0, 2), (1, 2))):
            x = index * 341
            draw.text((x + 4, row_top + 20), "".join("XYZ"[axis] for axis in axes) + " world point cloud", fill="white")
            canvas.paste(_orthographic(cloud, rgb, view_bounds, axes, current_tcp, action_tcp,
                                       role_points, role_valid, region_bounds=roi,
                                       current_label=current_label, chunk=chunk), (x, row_top + 38))
    if action_tcp is None:
        draw.text((8, ortho_top + 300), "GT goal unavailable: no goal marker, local cube or GT-centered refine views are generated.", fill=ACTION)
    else:
        draw.text((8, ortho_top + 300), "Magenta box: local cube footprint. Local views select observed points before global sampling.", fill="white")
        if not local_count:
            draw.text((8, local_top + 300), "No observed points in local cube; the GT keypoint does not create geometry.", fill="white")
        else:
            draw.text((8, local_top + 300), "Local cube is fixed at the GT keypoint; it is not a coarse prediction or the model's noisy training crop.", fill="white")
    if chunk is not None:
        from ..chunk_visualization import path_panel
        draw.text((8, chunk_top + 2), 'GT future chunk zoom (world metres): lines connect measured TCP samples; no planned intermediate path', fill=CHUNK)
        canvas.paste(path_panel(None, chunk['actions'], target_color=CHUNK), (8, chunk_top + 20))
    if object_overlay is not None:
        draw.text((8, object_top), f"Approximate object crops below | source={object_overlay['source']} | "
                  f"selected={len(object_overlay['boxes'])} | white=overlapping boxes; no instance masks", fill="white")
        if not object_overlay["boxes"]:
            draw.text((8, object_top + 24), "No objects selected in this frame; no object surfaces are generated.", fill="white")
        for index, box in enumerate(object_overlay["boxes"]):
            row_top = object_top + 24 + index * 318
            draw.text((8, row_top + 2), f"{_object_label(box)} | source={box['source_points']} "
                      f"world={box['world_points']} local={box['local_points']} | full-size world-aligned crop", fill=box["color"])
            overlay_view = _object_view(dict(object_overlay, boxes=[box]), box["bounds"])
            for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
                x = panel * 341
                draw.text((x + 4, row_top + 20), "".join("XYZ"[axis] for axis in axes) + " object cloud (world m)", fill="white")
                canvas.paste(_orthographic(np.empty((0,3)), np.empty((0,3), dtype=np.uint8), box["bounds"],
                                           axes, None, None, None, None, object_overlay=overlay_view), (x, row_top + 38))
            if not box["source_points"]:
                draw.text((8, row_top + 300), "Empty crop: no observed points; box and center do not create an object surface.", fill="white")
    draw.text((8, footer_top), "CPU diagnostics, not BridgeVLA renderer outputs. 1px points; stored XYZ unchanged. Black depth = invalid.", fill="white")
    if object_overlay is not None:
        draw.text((8, footer_top + 18), "Objects: approximate box crops, not instance/role labels. RGB/depth panels and saved observations unchanged.", fill="white")
    elif color_by_camera:
        for index, camera in enumerate(cameras):
            draw.text((8 + index * 204, footer_top + 18), camera, fill=CAMERA_COLORS[camera])
    else:
        draw.text((8, footer_top + 18), "Role masks: 30% RGB + 70% color. Site outlines are projected region points; visibility is not verified.", fill="white")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return path


class PreviewWriter:
    """Every N emitted samples per episode; 0 disables all output."""

    def __init__(self, output, every=0):
        if not isinstance(every, int) or isinstance(every, bool) or every < 0:
            raise ValueError("visualize_every must be a nonnegative integer")
        self.output, self.every, self.counts = Path(output), every, {}

    def write(self, observation, config, sample, **kwargs):
        if not self.every:
            return None
        episode = (sample["task"], sample["episode_index"])
        index = self.counts.get(episode, 0)
        self.counts[episode] = index + 1
        if index % self.every:
            return None
        return save_preview(inside(self.output, sample["id"] + "/preview.png"), observation, config, sample, **kwargs)
