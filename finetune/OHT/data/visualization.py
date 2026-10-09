"""CPU-only RGB-D/world geometry previews, separate from policy observations."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import inside


TARGET = np.array([35, 220, 75], dtype=np.uint8)
REFERENCE = np.array([65, 125, 255], dtype=np.uint8)
OVERLAP = np.array([255, 220, 50], dtype=np.uint8)
CURRENT = (0, 220, 255)
ACTION = (255, 70, 220)
PANEL_W, PANEL_H = 256, 192


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
        draw.text((x + 9, y - 16 if label == "TCP" else y + 6), label, fill=color)


def _camera_rgb(observation, camera, current_tcp, action_tcp, masks, specs, role_points, role_valid):
    raw = observation[f"{camera}_rgb"].transpose(1, 2, 0)
    image = Image.fromarray(role_overlay(raw, masks)).resize((PANEL_W, PANEL_H), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    height, width = raw.shape[:2]
    scale = np.array([PANEL_W / width, PANEL_H / height])
    K, transform = observation[f"{camera}_camera_intrinsics"], observation[f"{camera}_camera_extrinsics"]
    # Site regions are projected point hints, never painted as visible masks.
    if specs is not None:
        for index, (role, color) in enumerate((("target", TARGET), ("reference", REFERENCE))):
            if specs[role]["source"] == "site_region" and role_valid[index]:
                xy, visible = project_world(role_points[index], K, transform, (height, width))
                for point in xy[visible][::max(1, len(xy) // 128)]:
                    x, y = point * scale
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=tuple(map(int, color)))
    for pose, color, label in ((current_tcp, CURRENT, "TCP"), (action_tcp, ACTION, "goal")):
        if pose is not None:
            xy, visible = project_world(np.asarray(pose)[:3], K, transform, (height, width))
            if visible[0]:
                _mark(draw, xy[0] * scale, color, label)
    return image


def _orthographic(points, colors, bounds, axes, current_tcp, action_tcp, role_points, role_valid):
    width, height = 340, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    bounds = np.asarray(bounds)
    lower, span = bounds[:3], bounds[3:] - bounds[:3]
    horizontal, vertical = axes

    def pixels(cloud):
        uv = (np.asarray(cloud) - lower) / span
        return np.stack((uv[:, horizontal] * (width - 1),
                         (1 - uv[:, vertical]) * (height - 1)), axis=-1)

    if len(points):
        xy = pixels(points).astype(int)
        normal = next(axis for axis in range(3) if axis not in axes)
        # Diagnostic z-buffer viewed from the positive normal axis.
        order = np.argsort(points[:, normal], kind="stable")[::-1]
        flat = xy[:, 1] * width + xy[:, 0]
        _, first = np.unique(flat[order], return_index=True)
        chosen = order[first]
        image[xy[chosen, 1], xy[chosen, 0]] = colors[chosen]
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    if role_points is not None:
        for index, color in enumerate((TARGET, REFERENCE)):
            if role_valid[index]:
                cloud = role_points[index]
                valid = np.isfinite(cloud).all(axis=1) & (cloud >= bounds[:3]).all(axis=1)
                valid &= (cloud < bounds[3:]).all(axis=1)
                for x, y in pixels(cloud[valid]):
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=tuple(map(int, color)))
    for pose, color, label in ((current_tcp, CURRENT, "TCP"), (action_tcp, ACTION, "goal")):
        if pose is not None:
            point = np.asarray(pose)[:3]
            if np.isfinite(point).all() and (point >= bounds[:3]).all() and (point < bounds[3:]).all():
                _mark(draw, pixels(point[None])[0], color, label)
    names = "XYZ"
    draw.text((4, 4), f"{names[horizontal]}: {bounds[horizontal]:.3f} .. {bounds[horizontal+3]:.3f} m", fill="white")
    draw.text((4, 20), f"{names[vertical]}: {bounds[vertical]:.3f} .. {bounds[vertical+3]:.3f} m (up)", fill="white")
    return result


def save_preview(path, observation, config, sample, current_tcp=None, role_specs=None,
                 role_masks=None, role_points=None, role_valid=None):
    """Save sampled diagnostics without changing cached tensors or labels."""
    cameras = list(config["cameras"])
    action_tcp = np.asarray(sample["labels"]["gripper_pose"])
    bounds = np.asarray(config["scene_bounds"])
    chunks, rgb_chunks = [], []
    for camera in cameras:
        cloud = observation[f"{camera}_point_cloud"].reshape(3, -1).T
        colors = observation[f"{camera}_rgb"].reshape(3, -1).T
        valid = np.isfinite(cloud).all(axis=1) & (cloud >= bounds[:3]).all(axis=1)
        valid &= (cloud < bounds[3:]).all(axis=1)
        chunks.append(cloud[valid])
        rgb_chunks.append(colors[valid])
    points, colors = np.concatenate(chunks), np.concatenate(rgb_chunks)
    if len(points) > 20000:
        indices = np.linspace(0, len(points) - 1, 20000).astype(int)
        points, colors = points[indices], colors[indices]
    row_height, top = PANEL_H + 44, 96
    ortho_top = top + ((len(cameras) + 1) // 2) * row_height
    canvas = Image.new("RGB", (1024, ortho_top + 330), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"OHT {sample['id']} | {sample['split']} | frame {sample['frame']} -> {sample['target_frame']}", fill="white")
    draw.text((8, 24), f"time {sample['timestamp']:.3f} s | {sample['goal']}", fill="white")
    current = "not cached" if current_tcp is None else np.array2string(np.asarray(current_tcp)[:3], precision=3)
    draw.text((8, 42), f"World TCP {current} -> {np.array2string(action_tcp[:3], precision=3)} m", fill="white")
    draw.text((8, 60), "Cyan=current TCP; magenta=future action; green=Target; blue=Reference; yellow=overlap", fill="white")
    if role_specs is not None:
        status = " | ".join(f"{role}: {spec['source']} present={spec['present']} known={spec['known']} geometry={bool(role_valid[i])}"
                            for i, (role, spec) in enumerate((('target', role_specs['target']), ('reference', role_specs['reference']))))
        draw.text((8, 78), status, fill="white")
    for index, camera in enumerate(cameras):
        x, y = (index % 2) * 512, top + (index // 2) * row_height
        masks = {role: values[camera] for role, values in (role_masks or {}).items()}
        draw.text((x + 4, y + 2), camera + (" RGB + roles" if role_specs is not None else " RGB"), fill="white")
        canvas.paste(_camera_rgb(observation, camera, current_tcp, action_tcp, masks,
                                role_specs, role_points, role_valid), (x, y + 20))
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
    for index, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        x = index * 341
        draw.text((x + 4, ortho_top + 2), "".join("XYZ"[axis] for axis in axes) + " world point cloud", fill="white")
        canvas.paste(_orthographic(points, colors, bounds, axes, current_tcp, action_tcp,
                                   role_points, role_valid), (x, ortho_top + 22))
    draw.text((8, ortho_top + 284), "CPU diagnostic projection, not the BridgeVLA renderer. Black depth pixels are invalid.", fill="white")
    draw.text((8, ortho_top + 302), "Role masks: 30% RGB + 70% color. Site outlines are projected region points; visibility is not verified.", fill="white")
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
        return save_preview(inside(self.output, sample["id"] + ".png"), observation, config, sample, **kwargs)
