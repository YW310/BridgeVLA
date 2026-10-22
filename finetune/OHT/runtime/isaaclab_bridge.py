"""IsaacLab sensors and physical target tracking, independent of VLA weights."""
import numpy as np
from scipy.spatial.transform import Rotation

from ..data.geometry import array, pose_matrix, transform_matrix
from .executor import execution_target
from .joint_executor import arm_ids, terminated, numpy_value, action_layout
from .isaaclab_runtime import controller_offset


def pose_wxyz_matrix(pose):
    value = array(pose, (7,), 'world pose xyz+wxyz')
    return pose_matrix(value[:3], value[3:][[1, 2, 3, 0]])


def osc_command(current_link, target_link, world_from_base, position_scale, orientation_scale,
                max_translation=.05, max_rotation=.15, gripper_open=1):
    """Physical residual -> raw OSC units; clipping precedes division by live scales."""
    base_from_world = np.linalg.inv(transform_matrix(world_from_base))
    current = base_from_world @ transform_matrix(current_link)
    target = base_from_world @ transform_matrix(target_link)
    delta = target[:3, 3] - current[:3, 3]
    rotvec = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    scales = []
    for value in (position_scale, orientation_scale):
        value = np.broadcast_to(np.asarray(value, np.float64), (3,))
        if not np.isfinite(value).all() or (np.abs(value) < 1e-9).any():
            raise ValueError('OSC scales must be finite and nonzero')
        scales.append(value)
    if not np.isfinite([max_translation, max_rotation]).all() or min(max_translation, max_rotation) <= 0:
        raise ValueError('Physical OSC clip limits must be finite and positive')
    if gripper_open not in (0, 1):
        raise ValueError('Expected binary open01 gripper')
    return np.r_[np.clip(delta, -max_translation, max_translation) / scales[0],
                 np.clip(rotvec, -max_rotation, max_rotation) / scales[1], 2 * gripper_open - 1].astype(np.float32)


class IsaacLabBridge:
    def __init__(self, reference, env, client, task, profile, cameras, *, max_substeps=1,
                 max_translation=.05, max_rotation=.15, joint_executor=None):
        if type(max_substeps) is not int or max_substeps < 1:
            raise ValueError('max_substeps must be positive')
        self.ref, self.env, self.client = reference, env, client
        self.robot, self.task, self.profile, self.cameras = env.scene['robot'], task, profile, cameras
        self.link_to_tcp = transform_matrix(client.info['raw_observation']['data_profiles'][profile]['link_to_tcp'])
        self.max_substeps, self.max_translation, self.max_rotation = max_substeps, max_translation, max_rotation
        self.joint_executor = joint_executor
        self.last_done = False
        self.last_target = None
        self.control_offset = np.eye(4)
        mode = client.action_config['mode']
        if 'joint' in mode and joint_executor is None:
            raise ValueError('Joint modes require a joint-position executor')
        if 'eef' in mode:
            term = env.action_manager.get_term('arm_action')
            if 'OperationalSpace' not in type(term).__name__:
                raise ValueError('EEF execution requires an OperationalSpace action term')
            types = getattr(getattr(term.cfg, 'controller_cfg', None), 'target_types', None)
            if types is not None and list(types) != ['pose_rel']:
                raise ValueError('This OSC adapter requires controller target_types=[pose_rel]')
            if getattr(term.cfg, 'task_frame_rel_path', None):
                raise ValueError('This OSC adapter requires the robot-root task frame')
            body = getattr(term.cfg, 'body_name', None)
            if body is not None and hasattr(reference, 'ee_link') and body != reference.ee_link:
                raise ValueError('OSC body_name must match the runtime ee_link')
            body_to_control = controller_offset(term.cfg, getattr(reference, 'quaternion_order', 'wxyz'))
            body_to_observation = transform_matrix(getattr(reference, 'body_to_observation', np.eye(4)))
            self.control_offset = np.linalg.inv(body_to_observation) @ body_to_control

    def _numpy(self, value):
        return self.ref.tensor_to_numpy(value, np.float32)

    def link_pose(self):
        position, rotation = self.ref.get_ee_pose_in_world_frame(self.env, self.robot)
        return np.r_[self._numpy(position).reshape(3), self._numpy(rotation).reshape(4)]

    def capture(self):
        # Render-only refresh after reset/step; never advance physics to obtain RGB.
        self.ref._replicator_render_tick(self.env)
        _, ids = arm_ids(self.env, self.client.action_config)
        tensors = dict(joint_positions=numpy_value(self.robot.data.joint_pos)[0, ids].astype(np.float32),
                       ee_link_pose=self.link_pose().astype(np.float32),
                       gripper_position=np.asarray(self.ref.get_online_gripper_physical_state(
                           self.env, self.robot), np.float32).reshape(1))
        for name, camera in self.cameras.items():
            # Both runtimes expose metric Z-depth and live OpenGL sensor poses.
            rgb = np.asarray(self.ref.camera_rgb(camera))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
                raise ValueError(f'{name}: expected uint8 RGB/RGBA')
            depth = np.asarray(self.ref.camera_depth(camera), np.float32).copy()
            depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
            tensors.update({f'{name}_rgb': rgb[..., :3].copy(),
                            f'{name}_depth': depth,
                            f'{name}_intrinsics': np.asarray(self.ref.camera_intrinsic(camera), np.float32),
                            f'{name}_pose': np.asarray(self.ref.camera_pose_ros(
                                camera, self.env, robot=self.robot), np.float32)})
        return tensors

    def _base(self):
        if callable(getattr(self.ref, 'world_from_base', None)):
            return transform_matrix(self.ref.world_from_base())
        position = self._numpy(self.robot.data.root_pos_w[0]).copy()
        origins = getattr(self.env.scene, 'env_origins', None)
        if origins is not None:
            position -= self._numpy(origins[0])
        return pose_wxyz_matrix(np.r_[position, self._numpy(self.robot.data.root_quat_w[0])])

    def measured_state(self, measured=None):
        if measured is None:
            _, ids = arm_ids(self.env, self.client.action_config)
            measured = dict(ee_link_pose=self.link_pose(), joint_positions=numpy_value(self.robot.data.joint_pos)[0, ids],
                            gripper_position=self.ref.get_online_gripper_physical_state(self.env, self.robot))
        tcp = pose_wxyz_matrix(measured['ee_link_pose']) @ self.link_to_tcp
        return dict(ee_pose_wxyz=np.asarray(measured['ee_link_pose']).tolist(),
                    tcp_pose_xyzw=np.r_[tcp[:3, 3], Rotation.from_matrix(tcp[:3, :3]).as_quat()].tolist(),
                    joint_positions=np.asarray(measured['joint_positions']).tolist(),
                    gripper_position=float(np.asarray(measured['gripper_position']).reshape(-1)[0]))

    def tracking_errors(self, state):
        target = np.asarray(self.last_target)
        if 'joint' in self.client.action_config['mode']:
            return dict(joint_max_rad=float(np.max(np.abs(target[:6] - state['joint_positions']))))
        pose = np.asarray(state['tcp_pose_xyzw'])
        return dict(translation_m=float(np.linalg.norm(target[:3] - pose[:3])),
                    rotation_rad=float((Rotation.from_quat(target[3:7]) * Rotation.from_quat(pose[3:7]).inv()).magnitude()))

    def execute(self, action, measured, after_step=None):
        self.last_done = False
        current_link = pose_wxyz_matrix(measured['ee_link_pose'])
        current_tcp_matrix = current_link @ self.link_to_tcp
        current_tcp = np.r_[current_tcp_matrix[:3, 3], Rotation.from_matrix(current_tcp_matrix[:3, :3]).as_quat()]
        target = execution_target(action, self.client.action_config, current_tcp=current_tcp,
                                  current_joints=measured['joint_positions'])
        self.last_target = target.tolist()
        if 'joint' in self.client.action_config['mode']:
            commands = self.joint_executor(target, after_step=after_step)
            self.last_done = getattr(self.joint_executor, 'last_done', False)
            return commands
        target_link = pose_matrix(target[:3], target[3:7]) @ np.linalg.inv(self.link_to_tcp)
        term = self.env.action_manager.get_term('arm_action')
        cfg = term.cfg
        commands = []
        for _ in range(self.max_substeps):
            command = osc_command(pose_wxyz_matrix(self.link_pose()) @ self.control_offset,
                                  target_link @ self.control_offset, self._base(),
                                  cfg.position_scale, cfg.orientation_scale,
                                  self.max_translation, self.max_rotation, int(target[-1]))
            if hasattr(self.env.action_manager, 'active_terms'):
                layout = action_layout(self.env)
                ordered = np.empty(7, np.float32)
                ordered[layout['arm_action']] = command[:6]
                ordered[layout['gripper_action']] = command[-1]
            else:
                ordered = command
            result = self.env.step(self.ref.torch.as_tensor(ordered, device=self.env.device).reshape(1, 7))
            commands.append(ordered.tolist())
            self.last_done = terminated(result)
            stopped = after_step is not None and after_step(float(command[-1]))
            if stopped or self.last_done:
                break
        return commands
