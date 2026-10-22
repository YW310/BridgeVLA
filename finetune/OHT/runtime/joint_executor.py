"""BridgeVLA joint targets through the environment's existing position controller."""
import numpy as np
from ..data.geometry import array


def numpy_value(value):
    value = getattr(value, 'torch', value)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def arm_ids(env, options=None):
    term = env.action_manager.get_term('arm_action')
    indices = getattr(term, 'joint_ids', None)
    if indices is None:
        indices = getattr(term, '_joint_ids', None)
    robot = env.scene['robot']
    if isinstance(indices, slice):
        indices = list(range(len(robot.joint_names)))[indices]
    if indices is None:
        raise ValueError('arm_action must expose its joint IDs')
    indices = numpy_value(indices).astype(int).reshape(-1).tolist()
    if len(indices) != 6 or len(set(indices)) != 6:
        raise ValueError('arm_action must contain six distinct arm joints')
    names = (options or {}).get('joint_names')
    canonical = indices
    if names is not None:
        canonical = [list(robot.joint_names).index(name) for name in names]
        if set(canonical) != set(indices):
            raise ValueError('Checkpoint joint_names do not match the environment arm joints')
    return indices, canonical


def action_layout(env):
    manager = env.action_manager
    names, sizes = list(manager.active_terms), list(manager.action_term_dim)
    if set(names) != {'arm_action', 'gripper_action'} or len(names) != 2:
        raise ValueError('OHT executor requires only arm_action and gripper_action')
    result, start = {}, 0
    for name, size in zip(names, sizes):
        expected = 6 if name == 'arm_action' else 1
        if size != expected:
            raise ValueError(f'{name}: expected {expected} action dimensions')
        result[name] = slice(start, start + size)
        start += size
    return result


def terminated(result):
    """Gymnasium done is an episode boundary, never a generic success signal."""
    if isinstance(result, tuple) and len(result) == 5:
        return bool(numpy_value(result[2]).any() or numpy_value(result[3]).any())
    return False


class JointPositionExecutor:
    def __init__(self, env, torch_module, options, *, hold_steps=1, tolerance=.01, max_step=None):
        if type(hold_steps) is not int or hold_steps < 1:
            raise ValueError('joint hold_steps must be positive')
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('joint tolerance must be finite and nonnegative')
        if max_step is not None and (not np.isfinite(max_step) or max_step <= 0):
            raise ValueError('joint max_step must be finite and positive')
        self.env, self.torch, self.robot = env, torch_module, env.scene['robot']
        self.term = env.action_manager.get_term('arm_action')
        # Relative/velocity/effort action terms have different semantics.
        if type(self.term).__name__ != 'JointPositionAction':
            raise ValueError('Built-in joint executor needs JointPositionAction; select a joint-position env or custom executor')
        self.layout = action_layout(env)
        self.term_ids, self.canonical_ids = arm_ids(env, options)
        self.order = [self.canonical_ids.index(index) for index in self.term_ids]
        self.hold_steps, self.tolerance, self.max_step = hold_steps, tolerance, max_step
        self.last_done, self.last_tracking_error = False, None

    def _affine(self, name):
        if not hasattr(self.term, name):
            raise ValueError('JointPositionAction scale/offset interface is unavailable in this IsaacLab version')
        value = numpy_value(getattr(self.term, name)).astype(np.float64)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        value = np.broadcast_to(value, (6,))
        if not np.isfinite(value).all():
            raise ValueError('Joint controller scale/offset must be finite')
        return value

    def step(self, absolute_target):
        """Apply exactly one control step, for fixed-target waypoint tracking."""
        target = array(absolute_target, (7,), 'absolute joint target')
        if target[-1] not in (0, 1):
            raise ValueError('Expected open01 gripper')
        desired = target[:6][self.order]
        limits = getattr(self.robot.data, 'soft_joint_pos_limits', None)
        if limits is not None:
            limits = numpy_value(limits)[0, self.term_ids]
            if ((desired < limits[:, 0]) | (desired > limits[:, 1])).any():
                raise ValueError('Predicted joint target is outside the configured soft joint limits')
        scale, offset = self._affine('_scale'), self._affine('_offset')
        if (np.abs(scale) < 1e-9).any():
            raise ValueError('Joint controller scale must be nonzero')
        current = numpy_value(self.robot.data.joint_pos)[0, self.term_ids]
        commanded = desired if self.max_step is None else current + np.clip(desired - current, -self.max_step, self.max_step)
        raw = (commanded - offset) / scale
        previous = self.term.raw_actions.clone()
        try:
            self.term.process_actions(self.torch.as_tensor(raw, device=self.env.device, dtype=self.torch.float32)[None])
            if not np.allclose(numpy_value(self.term.processed_actions)[0], commanded, atol=1e-5, rtol=1e-5):
                raise ValueError('Joint action transform clips or changes the requested physical target')
        finally:
            self.term.process_actions(previous)
        command = np.empty(7, np.float32)
        command[self.layout['arm_action']] = raw
        command[self.layout['gripper_action']] = 2 * target[-1] - 1
        result = self.env.step(self.torch.as_tensor(command, device=self.env.device)[None])
        self.last_done = terminated(result)
        current = numpy_value(self.robot.data.joint_pos)[0, self.term_ids]
        self.last_tracking_error = float(np.max(np.abs(current - desired)))
        return command.tolist()

    def __call__(self, absolute_target, after_step=None):
        target = array(absolute_target, (7,), 'absolute joint target')
        commands, self.last_done = [], False
        for _ in range(self.hold_steps):
            commands.append(self.step(target))
            stopped = after_step is not None and after_step(float(2 * target[-1] - 1))
            if stopped or self.last_done or self.last_tracking_error <= self.tolerance:
                break
        return commands
