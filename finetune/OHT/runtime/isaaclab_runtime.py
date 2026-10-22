"""BridgeVLA adapter for an existing IsaacLab env and registered Camera sensors."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from .joint_executor import numpy_value
from ..data.geometry import pose_matrix


def controller_offset(cfg, quaternion_order='wxyz'):
    offset = getattr(cfg, 'body_offset', None)
    if offset is None:
        return np.eye(4)
    rotation = np.asarray(offset.rot)
    if quaternion_order == 'wxyz':
        rotation = rotation[[1, 2, 3, 0]]
    return pose_matrix(offset.pos, rotation)


class IsaacLabRuntime:
    def __init__(self, env, cameras, *, ee_link, quaternion_order, simulation_app,
                 owns_env=True, owns_app=False, ee_pose_source='body'):
        import torch
        if quaternion_order not in ('wxyz', 'xyzw'):
            raise ValueError('Declare the installed IsaacLab quaternion order explicitly')
        if ee_pose_source not in ('body', 'controller'):
            raise ValueError('ee_pose_source must be body or controller')
        if getattr(env, 'num_envs', 1) != 1:
            raise ValueError('BridgeVLA HTTP runtime supports one environment')
        self.env, self.cameras, self.simulation_app = env, cameras, simulation_app
        for camera in cameras.values():
            cfg = getattr(camera, 'cfg', None)
            if cfg is not None and hasattr(cfg, 'update_latest_camera_pose'):
                # CameraCfg defaults to returning the initialization pose.
                # Force live poses before the first render/data refresh.
                cfg.update_latest_camera_pose = True
        self.quaternion_order, self.torch = quaternion_order, torch
        self.owns_env, self.owns_app = owns_env, owns_app
        self.args_cli = SimpleNamespace(device=env.device)
        robot = env.scene['robot']
        if ee_link not in robot.body_names:
            raise ValueError(f'Unknown EE link: {ee_link}')
        self.ee_link = ee_link
        self.body_index = list(robot.body_names).index(ee_link)
        self.ee_pose_source = ee_pose_source
        cfg = getattr(env.action_manager.get_term('arm_action'), 'cfg', None)
        if ee_pose_source == 'controller' and getattr(cfg, 'body_name', ee_link) != ee_link:
            raise ValueError('Controller EE pose source requires ee_link to match arm_action.body_name')
        self.body_to_observation = controller_offset(cfg, quaternion_order) if ee_pose_source == 'controller' else np.eye(4)

    @staticmethod
    def tensor_to_numpy(value, dtype=np.float32):
        return np.asarray(numpy_value(value), dtype=dtype)

    def _origin(self):
        origins = getattr(self.env.scene, 'env_origins', None)
        return np.zeros(3) if origins is None else numpy_value(origins)[0]

    def _wxyz(self, value):
        value = np.asarray(value)
        return value if self.quaternion_order == 'wxyz' else value[[3, 0, 1, 2]]

    def world_from_base(self):
        robot = self.env.scene['robot']
        position = numpy_value(robot.data.root_pos_w)[0] - self._origin()
        rotation = self._wxyz(numpy_value(robot.data.root_quat_w)[0])
        return pose_matrix(position, rotation[[1, 2, 3, 0]])

    def get_ee_pose_in_world_frame(self, env, robot):
        position = numpy_value(robot.data.body_pos_w)[0, self.body_index] - self._origin()
        rotation = self._wxyz(numpy_value(robot.data.body_quat_w)[0, self.body_index])
        measured = pose_matrix(position, rotation[[1, 2, 3, 0]]) @ self.body_to_observation
        return measured[:3, 3], Rotation.from_matrix(measured[:3, :3]).as_quat()[[3, 0, 1, 2]]

    @staticmethod
    def get_online_gripper_physical_state(env, robot):
        term = env.action_manager.get_term('gripper_action')
        ids = getattr(term, 'joint_ids', None)
        if ids is None:
            ids = getattr(term, '_joint_ids', None)
        if ids is None:
            raise ValueError('gripper_action must expose measured joint IDs')
        if not isinstance(ids, slice):
            ids = numpy_value(ids).astype(int)
        return np.array([numpy_value(robot.data.joint_pos)[0, ids].sum()], np.float32)

    def _replicator_render_tick(self, env):
        env.sim.render()
        for camera in self.cameras.values():
            camera.update(0., force_recompute=True)

    @staticmethod
    def camera_rgb(camera):
        return numpy_value(camera.data.output['rgb'])[0]

    @staticmethod
    def camera_depth(camera):
        depth = numpy_value(camera.data.output['distance_to_image_plane'])[0]
        return depth[..., 0] if depth.ndim == 3 and depth.shape[-1] == 1 else depth

    @staticmethod
    def camera_intrinsic(camera):
        return numpy_value(camera.data.intrinsic_matrices)[0]

    def camera_pose_ros(self, camera, env, robot=None):
        # Kept as a compatibility method name; the raw API explicitly uses OpenGL axes.
        position = numpy_value(camera.data.pos_w)[0] - self._origin()
        rotation = self._wxyz(numpy_value(camera.data.quat_w_opengl)[0])
        return np.r_[position, rotation]

    def close(self):
        try:
            if self.owns_env:
                self.env.close()
        finally:
            if self.owns_app:
                self.simulation_app.close()
