# Copy from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/utils/custom_rlbench_env.py
from bridgevla.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
from bridgevla.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from yarr.utils.process_str import change_case
import numpy as np


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        self._oracle_provider = kwargs.pop('oracle_provider', None)
        self._oracle_provider_ready = False
        self._oracle_episode_token = -1
        self._oracle_demo = None
        self._oracle_demo_index = None
        self._oracle_ground_truth_frames = None
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    @property
    def oracle_provider(self):
        return self._oracle_provider

    @property
    def observation_elements(self):
        # Masks are captured only for the provider and are not policy inputs.
        return [
            element for element in super().observation_elements
            if not element.name.endswith('_mask')
        ]

    def _oracle_task_name(self):
        return change_case(self._task._task.__class__.__name__)

    def _oracle_variation(self):
        task = self._task._task
        for owner, attribute in (
            (self._task, '_variation_number'),
            (task, '_variation_index'),
            (task, 'variation_index'),
            (task, 'target_shelf'),
            (task, '_index'),
        ):
            value = getattr(owner, attribute, None)
            if value is not None:
                return int(value)
        return 0

    def _reset_oracle_provider(self, variation=None):
        if self._oracle_provider is None:
            return
        if variation is None:
            variation = self._oracle_variation()
        self._oracle_provider.reset(
            self._task,
            self._oracle_task_name(),
            int(variation),
            int(self._oracle_episode_token),
        )
        self._oracle_provider_ready = True

    def extract_obs(self, obs, *args, **kwargs):
        if self._oracle_provider is None:
            return super().extract_obs(obs, *args, **kwargs)
        if not self._oracle_provider_ready:
            self._reset_oracle_provider()
        mask_values = {}
        for camera in self._oracle_provider.cameras:
            key = f'{camera}_mask'
            mask_values[key] = getattr(obs, key, None)
            setattr(obs, key, None)
        try:
            extracted = super().extract_obs(obs, *args, **kwargs)
        finally:
            for key, value in mask_values.items():
                setattr(obs, key, value)
        return self._oracle_provider.enrich(obs, extracted)

    def reset(self) -> dict:
        self._oracle_episode_token = self._episode_index
        self._oracle_provider_ready = False
        self._oracle_demo = None
        self._oracle_demo_index = None
        self._oracle_ground_truth_frames = None
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]
        self._oracle_demo = d
        self._oracle_demo_index = int(i)
        self._oracle_ground_truth_frames = None

        self._oracle_episode_token = int(i)
        self._oracle_provider_ready = False
        self._reset_oracle_provider(d.variation_number)
        if self._oracle_provider is not None:
            self._oracle_provider.set_sample_frame(0)
        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict

    def get_ground_truth_action(self, episode_idx):
        """Return keypoint expert actions and expose their raw demo frames."""
        if self._oracle_demo is None or self._oracle_demo_index != int(episode_idx):
            self._oracle_demo = self._task.get_demos(
                1,
                live_demos=False,
                random_selection=False,
                from_episode_number=int(episode_idx),
            )[0]
            self._oracle_demo_index = int(episode_idx)
        frames = list(keypoint_discovery(self._oracle_demo))
        self._oracle_ground_truth_frames = frames
        if self._oracle_provider is not None:
            self._oracle_provider.set_expected_sample_frames(frames)
        return [
            np.concatenate(
                (
                    np.asarray(self._oracle_demo[frame].gripper_pose),
                    np.asarray([self._oracle_demo[frame].gripper_open]),
                )
            ).astype(np.float32, copy=False)
            for frame in frames
        ]

    def step(self, act_result):
        if self._oracle_provider is not None:
            sample_frame = None
            if (
                self._oracle_ground_truth_frames is not None
                and self._i < len(self._oracle_ground_truth_frames)
            ):
                sample_frame = self._oracle_ground_truth_frames[self._i]
            self._oracle_provider.set_sample_frame(sample_frame)
        return super().step(act_result)
