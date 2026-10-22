"""Replay the recorded training endpoints to test waypoint execution, without a policy."""
import numpy as np

from ..data.actions import keypoint_options, keypoints
from ..data.reader import read_episode
from .trajectory_replay import TrajectoryReplay


class KeypointReplay(TrajectoryReplay):
    """The original episode's actual configured keypoints, with no invented path.

    Endpoint tracking uses variable simulator time. Source frame timestamps are
    retained for diagnostics, not used to claim full-rate command reproduction.
    Joint/object initialization and TCP calibration are inherited unchanged.
    """

    def __init__(self, args):
        super().__init__(args)
        columns = read_episode(self.root, self.record)
        options = keypoint_options(self.config.get('keypoints'))
        all_indices = keypoints(self.poses, self.targets[:, -1], columns['instruction_id'],
                               timestamps=self.timestamps, **options)
        self.keyframe_indices = [frame for frame in all_indices if frame > self.start]
        if not self.keyframe_indices:
            raise ValueError('No future recorded keypoints after replay-start-frame; '
                             'the configured extraction does not invent a terminal goal')
        self.length = len(self.keyframe_indices)
        intervals = np.diff(self.timestamps)
        self.metadata.update(
            source='observed_future_keypoint_tcp', available_actions=self.length,
            keypoints=dict(options, indices=list(self.keyframe_indices), all_indices=list(all_indices)),
            source_frame_interval=dict(min_seconds=float(intervals.min()), max_seconds=float(intervals.max()),
                                       count=int(len(intervals))),
            timing='variable waypoint tracking; source timestamps are provenance only',
            path_source='endpoint targets only; recorded intermediate motion is not supplied',
        )

    def validate_runtime(self, env, args):
        if args.execution_mode != 'waypoint':
            raise ValueError('Keypoint replay requires waypoint execution')
        if type(args.max_steps) is not int or args.max_steps < 1:
            raise ValueError('Keypoint replay max_steps must be a positive bounded integer')
        if not np.isfinite(float(env.step_dt)) or float(env.step_dt) <= 0:
            raise ValueError('Simulator step_dt must be finite and positive')

    def action(self, step):
        if type(step) is not int or not 0 <= step < self.length:
            raise IndexError('Keypoint trajectory exhausted')
        source = self.start if step == 0 else self.keyframe_indices[step - 1]
        target = self.keyframe_indices[step]
        return self.targets[target].copy(), dict(
            source_frame=source, target_frame=target,
            source_timestamp=float(self.timestamps[source]), target_timestamp=float(self.timestamps[target]),
            target_schedule='keypoints', keypoint_method=self.metadata['keypoints']['method'],
        )
