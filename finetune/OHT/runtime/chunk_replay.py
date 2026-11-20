"""Recorded future TCP chunks through the same executor as a chunk policy."""
import numpy as np

from .trajectory_replay import TrajectoryReplay
from .chunk_protocol import chunk_spec, validate_spec
from ..data.chunk_dataset import load_chunk_contract


class ChunkReplay(TrajectoryReplay):
    def __init__(self, args):
        if not args.replay_contract:
            raise ValueError('GT chunk replay requires --replay-contract with a bound V4 replay')
        contract = load_chunk_contract(args.replay_contract)
        if contract['chunk']['frame_stride'] != 1:
            raise ValueError('GT chunk replay currently requires chunk.frame_stride=1')
        super().__init__(args)
        horizon = getattr(args, 'chunk_execute_horizon', None)
        horizon = 4 if horizon is None else horizon
        self.chunk_spec = validate_spec(chunk_spec(dict(horizon=contract['chunk']['horizon'],
            execute_horizon=horizon), contract))
        self.execute_horizon = horizon
        gripper_mode = getattr(args, 'replay_gripper_mode', 'recorded')
        if gripper_mode not in ('recorded', 'open'):
            raise ValueError('Unsupported GT chunk gripper mode')
        if gripper_mode == 'open':
            self.targets[:, -1] = 1.
        self.raw_transitions = self.length
        self.length = (self.raw_transitions + horizon - 1) // horizon
        self.info['chunk_spec'] = self.chunk_spec
        self.metadata.update(source='observed_future_tcp_chunk', available_actions=self.length,
            raw_transitions=self.raw_transitions, chunk_spec=self.chunk_spec,
            gripper_mode=gripper_mode,
            gripper_source=('forced open, arm-tracking diagnostic only' if gripper_mode == 'open' else
                            'recorded measured state with bound training calibration/hysteresis'),
            tail='Repeat last pose only for fixed K shape; execute valid raw ticks only')

    def validate_runtime(self, env, args):
        if args.execution_mode != 'chunk':
            raise ValueError('GT chunk replay requires chunk execution')
        if args.max_substeps != 1:
            raise ValueError('GT chunk replay requires --max-substeps 1')
        if type(args.max_steps) is not int or args.max_steps < 1:
            raise ValueError('GT chunk replay max_steps must be a positive bounded integer')
        end = self.start + min(self.raw_transitions, args.max_steps * self.execute_horizon)
        intervals = np.diff(self.timestamps)[self.start:end]
        if not np.isfinite(float(env.step_dt)) or not np.allclose(intervals, env.step_dt, atol=1e-5, rtol=1e-4):
            raise ValueError('GT chunk raw frame interval must match simulator step_dt')
        if not np.allclose(intervals, self.chunk_spec['action_dt'], atol=1e-5, rtol=1e-4):
            raise ValueError('GT chunk raw frame interval differs from replay contract action_dt')

    def action(self, step):
        if type(step) is not int or not 0 <= step < self.length:
            raise IndexError('Chunk trajectory exhausted')
        frame = self.start + step * self.execute_horizon
        count = min(self.execute_horizon, len(self.targets) - 1 - frame)
        frames = np.minimum(frame + np.arange(1, self.chunk_spec['horizon'] + 1), len(self.targets)-1)
        return self.targets[frames].copy(), dict(source_frame=frame, target_frame=frame+count,
            source_timestamp=float(self.timestamps[frame]), target_timestamp=float(self.timestamps[frame+count]),
            target_schedule='chunk', valid_execute_horizon=count,
            chunk_target_frames=frames.tolist())
