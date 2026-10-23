"""Existing OHT USD cameras captured without importing the reference policy client."""
import numpy as np
from scipy.spatial.transform import Rotation

from ..data.geometry import pose_matrix, transform_matrix
from .isaaclab_runtime import IsaacLabRuntime
from .joint_executor import numpy_value


def annotator_array(data):
    if isinstance(data, dict):
        data = data.get('data')
    if data is None:
        raise ValueError('Camera annotator returned no data; check headless rendering and camera warmup')
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(numpy_value(data))


def projection_intrinsics(params, width, height):
    if not isinstance(params, dict) or 'cameraProjection' not in params:
        raise ValueError('camera_params must provide cameraProjection')
    model = params.get('cameraModel', 'pinhole')
    if model not in (0, 'pinhole'):
        raise ValueError('BridgeVLA RGB-D backprojection requires a pinhole camera')
    size = np.asarray(params.get('renderProductResolution', [width, height])).reshape(-1)
    if size.shape != (2,) or not np.array_equal(size, [width, height]):
        raise ValueError('Camera projection resolution differs from captured RGB-D')
    projection = np.asarray(params['cameraProjection'], np.float64).reshape(4, 4)
    matrix = np.array([[abs(projection[0, 0]) * width / 2, 0, (1 - projection[0, 2]) * width / 2],
                       [0, abs(projection[1, 1]) * height / 2, (1 + projection[1, 2]) * height / 2],
                       [0, 0, 1]], np.float32)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError('Invalid pinhole projection from camera_params')
    return matrix


def render_camera_matrix(params, metres_per_unit):
    """Recover render-time OpenGL camera->world in metres.

    Replicator cameraViewTransform multiplies row vectors from world to camera:
    its published example for a camera at scene X=100 has -100 in row four.
    Transpose BEFORE inverting to obtain our column-vector camera->world frame.
    The view translation is in scene units; Z-depth is already in metres.
    See NVIDIA's CameraParams annotator example, not the ambiguously worded
    camera-to-world description in some versions of its API documentation.
    """
    if not isinstance(params, dict) or params.get('cameraViewTransform') is None:
        raise ValueError('camera_params has no cameraViewTransform')
    units = float(metres_per_unit)
    reported_units = float(params.get('metersPerSceneUnit', units))
    if (not np.isfinite([units, reported_units]).all() or min(units, reported_units) <= 0 or
            not np.isclose(units, reported_units, rtol=1e-5, atol=1e-10)):
        raise ValueError('camera_params scene units disagree with the USD stage')
    view = np.asarray(params['cameraViewTransform'], np.float64)
    if view.size != 16:
        raise ValueError('cameraViewTransform must contain 16 values')
    # Reject singular, reflected, scaled and non-affine matrices before they can
    # masquerade as a valid Rotation.from_matrix result. Never guess axes from
    # how near the pose is to the robot or hard-code an orientation correction.
    view = transform_matrix(view.reshape(4, 4).T, 'cameraViewTransform')
    world_from_camera = np.linalg.inv(view)
    world_from_camera[:3, 3] *= reported_units
    return transform_matrix(world_from_camera, 'render camera pose')


def select_camera_paths(prims, mapping):
    """Reject ambiguous names instead of silently binding a different camera."""
    result = {}
    for name, selected in mapping.items():
        matches = [path for path in prims if path == selected] if selected.startswith('/') else [
            path for path in prims if path.rsplit('/', 1)[-1] == selected]
        if len(matches) != 1:
            raise ValueError(f'{name}: camera {selected!r} resolved to {matches}; use an exact USD path in --camera-map')
        result[name] = matches[0]
    if len(set(result.values())) != len(result):
        raise ValueError('Checkpoint cameras must bind to distinct USD prims')
    return result


def usd_rigid_matrix(prim, Usd, UsdGeom, metres_per_unit):
    # Gf uses row-vector transforms; the BridgeVLA geometry code uses column vectors.
    raw = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), np.float64).T
    if raw.shape != (4, 4) or not np.isfinite(raw).all():
        raise ValueError('Invalid USD camera/body transform')
    # Preserve world translation, remove authored mesh/asset scale from frame axes.
    u, scales, vt = np.linalg.svd(raw[:3, :3])
    rotation = u @ vt
    if (scales <= 1e-12).any() or np.linalg.det(rotation) <= 0:
        raise ValueError('Singular or reflected USD camera/body transform')
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, raw[:3, 3] * metres_per_unit
    return result


def report_cleanup(errors):
    invalid = [(where, exc) for where, exc in errors
               if ' detach ' in where and isinstance(exc, TypeError) and 'Invalid NodeObj' in str(exc)]
    if invalid:
        print(f'[BridgeVLA] camera cleanup: skipped {len(invalid)} already-invalid OmniGraph annotator handles', flush=True)
    for where, exc in errors:
        if not (' detach ' in where and isinstance(exc, TypeError) and 'Invalid NodeObj' in str(exc)):
            print(f'[BridgeVLA] camera cleanup warning: {where}: {type(exc).__name__}: {exc}', flush=True)


class ReplicatorCamera:
    def __init__(self, name, prim, env, rep, Usd, UsdGeom, metres_per_unit, width, height):
        self.name, self.prim, self.env = name, prim, env
        self.Usd, self.UsdGeom, self.metres_per_unit = Usd, UsdGeom, metres_per_unit
        self.width, self.height = width, height
        self.product, self.annotators = None, {}
        self.body_index, self.body_from_camera = None, None
        self.pose_source, self.pose_diagnostics = None, {}
        if str(UsdGeom.Camera(prim).GetProjectionAttr().Get()) != 'perspective':
            raise ValueError(f'{name}: only perspective USD cameras are supported')
        robot = env.scene['robot']
        body_names = list(robot.body_names)
        parent = prim.GetParent()
        while parent.IsValid():
            if parent.GetName() in body_names:
                self.body_index = body_names.index(parent.GetName())
                self.body_from_camera = np.linalg.inv(self.usd_matrix(parent)) @ self.usd_matrix(prim)
                break
            parent = parent.GetParent()
        if name == 'wrist' and self.body_index is None:
            raise ValueError('Wrist camera must have a robot body ancestor for live physics pose tracking')
        try:
            self.product = rep.create.render_product(str(prim.GetPath()), (width, height))
            for kind in ('rgb', 'distance_to_image_plane', 'camera_params'):
                annotator = rep.AnnotatorRegistry.get_annotator(kind)
                self.annotators[kind] = annotator
                annotator.attach(self.product)
        except BaseException:
            self.close()
            raise

    def usd_matrix(self, prim):
        return usd_rigid_matrix(prim, self.Usd, self.UsdGeom, self.metres_per_unit)

    def rgb(self):
        value = annotator_array(self.annotators['rgb'].get_data())
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[:2] != (self.height, self.width) or value.shape[2] not in (3, 4):
            raise ValueError(f'{self.name}: expected uint8 RGB/RGBA at configured resolution')
        return np.ascontiguousarray(value[..., :3])

    def depth(self):
        value = annotator_array(self.annotators['distance_to_image_plane'].get_data())
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.shape != (self.height, self.width):
            raise ValueError(f'{self.name}: expected metric Z-depth at configured resolution')
        return value.astype(np.float32, copy=True)

    def intrinsic(self):
        return projection_intrinsics(self.annotators['camera_params'].get_data(), self.width, self.height)

    def pose(self, runtime):
        # Use the same render-time extrinsic family as recorded OHT observations.
        # Static USD transforms can disagree with the actual rendered camera,
        # and a valid wrist view already contains its complete mounted offset.
        source, reason = 'camera_params', None
        try:
            annotator = getattr(self, 'annotators', {}).get('camera_params')
            params = None if annotator is None else annotator.get_data()
            world_from_camera = render_camera_matrix(params, getattr(self, 'metres_per_unit', 1.))
        except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError, RuntimeError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
            if self.body_index is None:
                source = 'usd'
                world_from_camera = self.usd_matrix(self.prim)
            else:
                source = 'physics_offset'
                data = runtime.env.scene['robot'].data
                position = numpy_value(data.body_pos_w)[0, self.body_index]
                rotation = runtime._wxyz(numpy_value(data.body_quat_w)[0, self.body_index])
                world_from_camera = pose_matrix(position, rotation[[1, 2, 3, 0]]) @ self.body_from_camera
        position = world_from_camera[:3, 3] - runtime._origin()
        quaternion = Rotation.from_matrix(world_from_camera[:3, :3]).as_quat()[[3, 0, 1, 2]]
        pose = np.r_[position, quaternion].astype(np.float32)
        previous_source = getattr(self, 'pose_source', None)
        self.pose_source = source
        self.pose_diagnostics = dict(source=source, fallback_reason=reason,
                                     camera_pose_wxyz=pose.tolist(),
                                     frame='env-origin-relative OpenGL camera-to-world')
        if previous_source != source and getattr(self, 'name', None):
            suffix = '' if reason is None else f' fallback={reason}'
            print(f'[BridgeVLA] camera {self.name} pose_source={source}{suffix}', flush=True)
        return pose

    def detach(self):
        """Release graph consumers while every camera's render product still exists."""
        errors = []
        annotators, self.annotators = self.annotators, {}
        for kind, annotator in annotators.items():
            try:
                annotator.detach(self.product)
            except Exception as exc:
                errors.append((f'{self.name} detach {kind}', exc))
        return errors

    def destroy_product(self):
        errors = []
        product, self.product = self.product, None
        if product is not None:
            try:
                product.destroy()
            except Exception as exc:
                errors.append((f'{self.name} destroy render product', exc))
        return errors

    def close(self):
        report_cleanup(self.detach() + self.destroy_product())


class ReplicatorCameras:
    def __init__(self, env, mapping, width, height):
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Usd, UsdGeom

        self.cameras = {}
        stage = omni.usd.get_context().get_stage()
        prims = {str(prim.GetPath()): prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)}
        # An explicit path can also select a camera inside an instance proxy.
        for value in mapping.values():
            if value.startswith('/'):
                prim = stage.GetPrimAtPath(value)
                if prim.IsValid() and prim.IsA(UsdGeom.Camera):
                    prims[value] = prim
        paths = select_camera_paths(prims, mapping)
        units = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not np.isfinite(units) or units <= 0:
            raise ValueError('USD stage must declare valid metres per unit')
        try:
            for name, path in paths.items():
                self.cameras[name] = ReplicatorCamera(name, prims[path], env, rep, Usd, UsdGeom, units, width, height)
                print(f'[BridgeVLA] camera {name} -> {path}', flush=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        # Replicator graph dependencies span cameras: do not destroy camera 1's
        # render product before detaching camera 2..N's annotators.
        errors = []
        for camera in self.cameras.values():
            errors.extend(camera.detach())
        for camera in self.cameras.values():
            errors.extend(camera.destroy_product())
        report_cleanup(errors)


class ReplicatorRuntime(IsaacLabRuntime):
    camera_source = 'usd'

    def __init__(self, env, manager, **kwargs):
        super().__init__(env, manager.cameras, **kwargs)
        self.manager = manager

    def _replicator_render_tick(self, env):
        env.sim.render()

    @staticmethod
    def camera_rgb(camera):
        return camera.rgb()

    @staticmethod
    def camera_depth(camera):
        return camera.depth()

    @staticmethod
    def camera_intrinsic(camera):
        return camera.intrinsic()

    def camera_pose_ros(self, camera, env, robot=None):
        return camera.pose(self)

    def close(self):
        try:
            self.manager.close()
        finally:
            super().close()
