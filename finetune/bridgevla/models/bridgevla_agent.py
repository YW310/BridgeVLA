'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/models/rvt_agent.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''

import math
import pprint
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
import RLBench.utils.peract_utils_rlbench as rlbench_utils
import GemBench.utils.peract_utils_gembench as gembench_utils
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con, aug_utils
from bridgevla.models.oracle_prior import (
    build_training_visualization_payload,
    choose_oracle_translation_loss,
    latest_replay_value,
    resolve_object_prior_mode,
    select_active_instance_points,
    select_relation_instance_points,
    valid_oracle_translation_loss,
    validate_oracle_prior_config,
)
from bridgevla.models.optimizer_utils import parameter_learning_rate
from bridgevla.models.object_conditioning import (
    active_semantic_target_mask,
    reference_null_loss,
    select_object_candidate_from_waypoint,
)
from yarr.agents.agent import ActResult
from PIL import Image, ImageDraw
import torch
import numpy as np
import os


def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    # Ensure data dimensions are correct
    assert points.shape[1] == 3 
    assert colors.shape[1] == 3
    
    # Automatically detect color value range and convert to 0-255
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)
    
    # Add keypoint (optional)
    if keypoint is not None:
        points = np.vstack([points, keypoint])
        colors = np.vstack([colors, np.array([255, 0, 0])])  # Mark keypoint in red

    # Write to PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, clr in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def visualize_images(
    color_tensor: torch.Tensor,
    heatmap_tensor: torch.Tensor,
    save_dir: str = "/opt/tiger/3D_OpenVLA/3d_policy/RVT/rvt_our/debug"
) -> None:
    """Save rendered views, the final heatmap, and its argmax overlay."""
    os.makedirs(save_dir, exist_ok=True)
    color_imgs = color_tensor.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    heatmaps = heatmap_tensor.detach().float().cpu().numpy()
    if heatmaps.ndim != 3 or heatmaps.shape[0] != color_imgs.shape[0]:
        raise ValueError(
            'heatmap_tensor must have shape [V, H, W] matching color_tensor'
        )
    for i in range(color_imgs.shape[0]):
        original_img = np.clip(color_imgs[i], 0, 1) * 255
        original_img = original_img.astype(np.uint8)
        Image.fromarray(original_img).save(os.path.join(save_dir, f"original_{i}.png"))
        normalized = _normalize_heatmap(heatmaps[i])
        gray_img = (normalized * 255).astype(np.uint8)
        Image.fromarray(gray_img, mode="L").save(os.path.join(save_dir, f"gray_{i}.png"))
        rgba = np.zeros((*original_img.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = original_img
        rgba[..., 3] = 77
        overlay_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        max_pos = np.unravel_index(normalized.argmax(), normalized.shape)
        x = max_pos[1]
        y = max_pos[0]
        point_radius = 5
        draw.ellipse(
            [x-point_radius, y-point_radius, x+point_radius, y+point_radius],
            fill=(255, 0, 0, 255)
        )
        overlay_img.save(os.path.join(save_dir, f"overlay_{i}.png"))


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(
        np.asarray(heatmap, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum <= minimum:
        return np.zeros_like(heatmap)
    return (heatmap - minimum) / (maximum - minimum)


def translation_heatmap_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert [V, H, W] translation logits to per-view probabilities."""
    if logits.ndim != 3:
        raise ValueError('translation logits must have shape [V, H, W]')
    views, height, width = logits.shape
    return torch.softmax(
        logits.float().reshape(views, height * width), dim=-1
    ).reshape(views, height, width)


def save_heatmap_views(
    heatmap_tensor: torch.Tensor,
    save_dir: str,
    prefix: str,
    color_tensor: torch.Tensor,
) -> None:
    """Save one grayscale map and red heatmap overlay for every MVT view."""
    os.makedirs(save_dir, exist_ok=True)
    if (
        heatmap_tensor.ndim == 3
        and heatmap_tensor.shape[-2:] != color_tensor.shape[-2:]
    ):
        heatmap_tensor = F.interpolate(
            heatmap_tensor[:, None].float(),
            size=color_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )[:, 0]
    heatmaps = heatmap_tensor.detach().float().cpu().numpy()
    color_imgs = color_tensor.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    if heatmaps.ndim != 3 or heatmaps.shape[0] != color_imgs.shape[0]:
        raise ValueError('heatmap and color view counts must match')
    for index in range(heatmaps.shape[0]):
        normalized = _normalize_heatmap(heatmaps[index])
        gray = (normalized * 255).astype(np.uint8)
        Image.fromarray(gray, mode='L').save(
            os.path.join(save_dir, f'{prefix}_{index}.png')
        )
        original = (np.clip(color_imgs[index], 0, 1) * 255).astype(np.float32)
        alpha = (0.65 * normalized)[..., None]
        red = np.zeros_like(original)
        red[..., 0] = 255
        overlay = np.clip(original * (1 - alpha) + red * alpha, 0, 255)
        Image.fromarray(overlay.astype(np.uint8)).save(
            os.path.join(save_dir, f'{prefix}_overlay_{index}.png')
        )


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gemma_lr: float = 0.0,
        gemma_layer_lr_decay: float = 1.0,
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = rlbench_utils.SCENE_BOUNDS,
        cameras: list = rlbench_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        oracle_prior_mode: str = 'none',
        object_prior_mode: str = 'none',
        object_prediction_confidence_threshold: float = 0.25,
        oracle_prior_sigma: float = 2.0,
        oracle_prior_active_role: str = 'auto',
        oracle_prior_strict: bool = False,
        oracle_prior_relation: bool = False,
        oracle_log_base_loss: bool = False,
        oracle_valid_only_loss: bool = False,
        object_slot_mask_loss_weight: float = 1.0,
        object_slot_null_loss_weight: float = 0.25,
        object_slot_diversity_loss_weight: float = 0.01,
        log_dir="",
    ):
        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self._gemma_lr = float(gemma_lr)
        self._gemma_layer_lr_decay = float(gemma_layer_lr_decay)
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        effective_prior_mode = resolve_object_prior_mode(
            object_prior_mode, oracle_prior_mode,
        )
        validate_oracle_prior_config(
            effective_prior_mode, oracle_prior_sigma, oracle_prior_active_role,
        )
        if not 0.0 <= object_prediction_confidence_threshold <= 1.0:
            raise ValueError(
                'object_prediction_confidence_threshold must be in [0, 1]'
            )
        if oracle_valid_only_loss and effective_prior_mode != 'o2_gt_instance':
            raise ValueError(
                'oracle_valid_only_loss requires O2 GT-instance mode'
            )
        self.object_prior_mode = effective_prior_mode
        # Legacy name retained for checkpoint/evaluation compatibility.
        self.oracle_prior_mode = effective_prior_mode
        self.object_prediction_confidence_threshold = float(
            object_prediction_confidence_threshold
        )
        self.oracle_prior_sigma = oracle_prior_sigma
        self.oracle_prior_active_role = oracle_prior_active_role
        self.oracle_prior_strict = oracle_prior_strict
        self.oracle_prior_relation = bool(oracle_prior_relation)
        self.oracle_log_base_loss = bool(oracle_log_base_loss)
        self.oracle_valid_only_loss = bool(oracle_valid_only_loss)
        slot_loss_weights = (
            object_slot_mask_loss_weight,
            object_slot_null_loss_weight,
            object_slot_diversity_loss_weight,
        )
        if any(weight < 0 for weight in slot_loss_weights):
            raise ValueError('object slot loss weights must be non-negative')
        self.object_slot_mask_loss_weight = float(object_slot_mask_loss_weight)
        self.object_slot_null_loss_weight = float(object_slot_null_loss_weight)
        self.object_slot_diversity_loss_weight = float(
            object_slot_diversity_loss_weight
        )
        self._oracle_missing_warning_shown = False
        # Runtime-only evaluation diagnostic. It is never read by update().
        self.heatmap_action_anchor = False
        self.bridgevla_aligned_objects = False
        self._heatmap_action_anchor_step = 0
        self._bridgevla_target_lock = -1
        self._bridgevla_last_gripper_open = None

        print("Cameras:",self.cameras)
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        trainable_parameters = [
            (name, parameter)
            for name, parameter in self._network.named_parameters()
            if parameter.requires_grad
        ]
        if self._gemma_lr > 0:
            num_gemma_layers = (
                self._net_mod.mvt1.model.config.text_config.num_hidden_layers
            )
            parameters_by_lr = {}
            for name, parameter in trainable_parameters:
                parameter_lr = parameter_learning_rate(
                    name,
                    self._lr,
                    self._gemma_lr,
                    self._gemma_layer_lr_decay,
                    num_gemma_layers,
                )
                parameters_by_lr.setdefault(parameter_lr, []).append(parameter)
            learning_rates = []
            if self._lr in parameters_by_lr:
                learning_rates.append(self._lr)
            learning_rates.extend(
                sorted(lr for lr in parameters_by_lr if lr != self._lr)
            )
            params_to_optimize = [
                {'params': parameters_by_lr[lr], 'lr': lr}
                for lr in learning_rates
            ]
            print(
                'Optimizer learning rates: '
                + ', '.join(f'{lr:.3e}' for lr in learning_rates)
            )
        else:
            params_to_optimize = [
                parameter for _, parameter in trainable_parameters
            ]

        optimizer_name = self._optimizer_type.lower()
        if optimizer_name == 'adam':
            optimizer_class = torch.optim.Adam
        elif optimizer_name == 'adamw':
            optimizer_class = torch.optim.AdamW
        else:
            raise ValueError(
                f'Unsupported optimizer_type: {self._optimizer_type}'
            )

        self._optimizer = optimizer_class(
            params_to_optimize,
            lr=self._lr,
            weight_decay=self._lambda_weight_l2,
        )

    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        self._optimizer.step()

    @property
    def oracle_prior_enabled(self):
        return self.oracle_prior_mode == 'o2_gt_instance'

    @property
    def predicted_object_prior_enabled(self):
        return self.object_prior_mode == 'o2_predicted_relation'

    @property
    def internal_object_slots_enabled(self):
        return self.object_prior_mode == 'o2_internal_slots'

    @property
    def object_prior_enabled(self):
        return (
            self.oracle_prior_mode == 'o2_gt_instance'
            or self.predicted_object_prior_enabled
            or self.internal_object_slots_enabled
        )

    def _select_oracle_prior_points(self, replay_sample, allow_missing=False):
        if not self.object_prior_enabled:
            return None, None, None
        if self.internal_object_slots_enabled and allow_missing:
            # Closed-loop inference is deliberately independent of Oracle
            # object fields. Slots infer both roles from the current features.
            return None, None, None
        if self.predicted_object_prior_enabled:
            roles = ('target', 'reference')
            required = []
            for role in roles:
                required.extend(
                    (
                        f'predicted_{role}_object_points',
                        f'predicted_{role}_object_valid',
                        f'predicted_{role}_present',
                        f'predicted_{role}_confidence',
                    )
                )
            missing = [key for key in required if key not in replay_sample]
            if missing:
                message = (
                    'Predicted-object input is missing required fields: '
                    + ', '.join(missing)
                )
                if allow_missing and not self.oracle_prior_strict:
                    if not self._oracle_missing_warning_shown:
                        print(
                            'WARNING: ' + message + ' Falling back to base '
                            'BridgeVLA features.', flush=True,
                        )
                        self._oracle_missing_warning_shown = True
                    return None, None, None
                raise KeyError(message)
            points = torch.stack(
                [
                    latest_replay_value(
                        replay_sample[f'predicted_{role}_object_points'], 3,
                    ).float()
                    for role in roles
                ],
                dim=1,
            )
            valid = torch.stack(
                [
                    latest_replay_value(
                        replay_sample[f'predicted_{role}_object_valid'], 1,
                    ).bool()
                    for role in roles
                ],
                dim=1,
            )
            present = torch.stack(
                [
                    latest_replay_value(
                        replay_sample[f'predicted_{role}_present'], 1,
                    ).bool()
                    for role in roles
                ],
                dim=1,
            )
            confidence = torch.stack(
                [
                    latest_replay_value(
                        replay_sample[f'predicted_{role}_confidence'], 1,
                    ).float()
                    for role in roles
                ],
                dim=1,
            )
            valid = (
                valid
                & present
                & torch.isfinite(confidence)
                & (confidence >= self.object_prediction_confidence_threshold)
            )
            # A semantically required but unavailable Reference is occluded or
            # ungrounded, not NULL. Disable the whole pair for that sample.
            unavailable_reference = present[:, 1] & ~valid[:, 1]
            valid[:, 0] &= ~unavailable_reference
            slots = torch.full(
                (points.shape[0], 2), -1,
                device=points.device, dtype=torch.long,
            )
            return points, valid, slots
        if self.oracle_prior_relation:
            pair_keys = (
                'oracle_target_object_points',
                'oracle_reference_object_points',
            )
            if any(key in replay_sample for key in pair_keys):
                missing_pair = [
                    key for key in pair_keys if key not in replay_sample
                ]
                if missing_pair:
                    raise KeyError(
                        'O2 relation input requires both Target and Reference: '
                        + ', '.join(missing_pair)
                    )
                role_points = [
                    latest_replay_value(replay_sample[key], 3).float()
                    for key in pair_keys
                ]
                points = torch.stack(role_points, dim=1)
                role_valid = []
                for role_name, role_points_value in zip(
                    ('target', 'reference'), role_points,
                ):
                    valid_value = replay_sample.get(
                        f'oracle_{role_name}_object_valid'
                    )
                    if valid_value is None:
                        valid_value = torch.ones(
                            role_points_value.shape[0],
                            device=role_points_value.device,
                            dtype=torch.bool,
                        )
                    else:
                        valid_value = latest_replay_value(
                            valid_value, 1,
                        ).bool()
                    role_valid.append(valid_value)
                valid = torch.stack(role_valid, dim=1)
                slots = torch.full(
                    (points.shape[0], 2), -1,
                    device=points.device, dtype=torch.long,
                )
                return points, valid, slots
            if 'oracle_active_object_points' in replay_sample:
                message = (
                    'O2 relation mode cannot infer a Target/Reference relation '
                    'from oracle_active_object_points; provide the full Oracle '
                    'object fields or both direct role point tensors.'
                )
                if allow_missing and not self.oracle_prior_strict:
                    if not self._oracle_missing_warning_shown:
                        print(
                            'WARNING: ' + message + ' Falling back to base '
                            'BridgeVLA features.', flush=True,
                        )
                        self._oracle_missing_warning_shown = True
                    return None, None, None
                raise KeyError(message)
        if 'oracle_active_object_points' in replay_sample:
            points = latest_replay_value(
                replay_sample['oracle_active_object_points'], 3,
            ).float()
            valid = replay_sample.get('oracle_active_object_valid')
            if valid is None:
                valid = torch.ones(
                    points.shape[0], device=points.device, dtype=torch.bool
                )
            else:
                valid = latest_replay_value(valid, 1).bool()
            slots = torch.full(
                (points.shape[0],), -1, device=points.device, dtype=torch.long
            )
            return points, valid, slots
        required = [
            'oracle_object_points', 'oracle_object_valid',
            'oracle_object_roles',
        ]
        if not self.oracle_prior_relation:
            required.append('low_dim_state')
        missing = [key for key in required if key not in replay_sample]
        if missing:
            if allow_missing and not self.oracle_prior_strict:
                if not self._oracle_missing_warning_shown:
                    print(
                        'WARNING: O2 Oracle fields are unavailable during act(); '
                        'falling back to base BridgeVLA features. Missing: '
                        + ', '.join(missing),
                        flush=True,
                    )
                    self._oracle_missing_warning_shown = True
                return None, None, None
            raise KeyError('O2 Oracle fields are missing: ' + ', '.join(missing))
        points = latest_replay_value(
            replay_sample['oracle_object_points'], 4,
        ).float()
        valid = latest_replay_value(
            replay_sample['oracle_object_valid'], 2,
        ).bool()
        roles = latest_replay_value(
            replay_sample['oracle_object_roles'], 2,
        ).long()
        if self.oracle_prior_relation:
            return select_relation_instance_points(
                points, valid, roles, strict=self.oracle_prior_strict,
            )
        low_dim = latest_replay_value(replay_sample['low_dim_state'], 2)
        return select_active_instance_points(
            points, valid, roles, gripper_open=low_dim[:, 0],
            active_role=self.oracle_prior_active_role,
            strict=self.oracle_prior_strict,
        )

    def _oracle_network_kwargs(self, points, valid, current_state=None):
        if points is None:
            if self.internal_object_slots_enabled:
                return {'current_state': current_state}
            return {}
        kwargs = {
            'oracle_prior_points': points,
            'oracle_prior_valid': valid,
            'oracle_prior_sigma': self.oracle_prior_sigma,
        }
        if current_state is not None:
            kwargs['current_state'] = current_state
        return kwargs

    def _object_slot_auxiliary_losses(self, output, oracle_valid, role_present=None,
                                     role_present_known=None):
        if not self.internal_object_slots_enabled:
            return {}
        if oracle_valid is None or oracle_valid.ndim != 2:
            raise ValueError('Internal slot training requires role validity [B,2]')
        stage_outputs = [output]
        if self.stage_two:
            stage_outputs.append(output['mvt2'])
        sums = {'mask': 0.0, 'presence': 0.0, 'diversity': 0.0}
        for stage_output in stage_outputs:
            required = (
                'object_slot_prior_logits',
                'object_slot_target_prior',
                'object_slot_masks',
                'object_slot_objectness_logits',
                'object_slot_reference_null_probability',
            )
            missing = [key for key in required if key not in stage_output]
            if missing:
                raise KeyError('Internal slot outputs are missing: ' + ', '.join(missing))
            logits = stage_output['object_slot_prior_logits']
            target = stage_output['object_slot_target_prior'].to(
                device=logits.device, dtype=logits.dtype,
            )
            batch_size, num_views, _, height, width = logits.shape
            target = F.interpolate(
                target.reshape(batch_size * num_views, 2, *target.shape[-2:]),
                size=(height, width),
                mode='area',
            ).view(batch_size, num_views, 2, height, width)
            mask_values = F.binary_cross_entropy_with_logits(
                logits, target, reduction='none',
            ).mean(dim=(1, 3, 4))
            valid = oracle_valid.to(device=logits.device).bool()
            mask_loss = (
                mask_values * valid.to(mask_values.dtype)
            ).sum() / valid.sum().clamp_min(1)

            objectness = stage_output['object_slot_objectness_logits']
            any_object_logit = torch.logsumexp(objectness, dim=1) - math.log(
                objectness.shape[1]
            )
            # Positive visible support teaches objectness. Unavailable geometry
            # is not a negative existence label (it may be an occluded object).
            objectness_values = F.binary_cross_entropy_with_logits(
                any_object_logit, torch.ones_like(any_object_logit), reduction='none',
            )
            support = valid[:, 0].to(objectness_values.dtype)
            target_objectness_loss = (
                objectness_values * support
            ).sum() / support.sum().clamp_min(1)
            null_loss = reference_null_loss(
                stage_output['object_slot_reference_null_probability'],
                role_present, role_present_known,
            )
            presence_loss = target_objectness_loss + null_loss

            masks = stage_output['object_slot_masks'].permute(0, 2, 1, 3, 4)
            masks = masks.flatten(2)
            normalized_masks = F.normalize(masks, dim=-1, eps=1e-6)
            overlap = torch.matmul(
                normalized_masks, normalized_masks.transpose(1, 2),
            )
            slot_count = overlap.shape[1]
            if slot_count > 1:
                identity = torch.eye(
                    slot_count, device=overlap.device, dtype=overlap.dtype,
                )[None]
                diversity_loss = (
                    overlap * (1.0 - identity)
                ).sum() / (batch_size * slot_count * (slot_count - 1))
            else:
                diversity_loss = overlap.new_zeros(())
            sums['mask'] = sums['mask'] + mask_loss
            sums['presence'] = sums['presence'] + presence_loss
            sums['diversity'] = sums['diversity'] + diversity_loss
        stage_count = len(stage_outputs)
        return {name: value / stage_count for name, value in sums.items()}

    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )


    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            if self.stage_two:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    def get_base_q_trans(self, out, dims):
        '''Return detached translation logits before the Oracle adapter.'''
        bs, nc, h, w = dims
        if 'trans_base' not in out:
            return None
        base = out['trans_base'].view(
            bs, nc, h * w,
        ).transpose(1, 2)
        if self.stage_two:
            stage_two_out = out['mvt2']
            if 'trans_base' not in stage_two_out:
                return None
            base2 = stage_two_out['trans_base'].view(
                bs, nc, h * w,
            ).transpose(1, 2)
            base = torch.cat((base, base2), dim=2)
        return base

    def get_base_q_rgc(self, out, batch_size):
        '''Return detached R/G/C logits before Oracle feature adaptation.'''
        if self.stage_two:
            out = out['mvt2']
        if self.rot_ver == 0:
            if 'feat_base' not in out:
                return None, None, None
            feat = out['feat_base'].view(batch_size, -1)
            rot_q = feat[:, :self.num_all_rot]
            grip_q = feat[:, self.num_all_rot:self.num_all_rot + 2]
            collision_q = feat[
                :, self.num_all_rot + 2:self.num_all_rot + 4
            ]
            return rot_q, grip_q, collision_q

        required = (
            'feat_x_base', 'feat_y_base', 'feat_z_base',
            'feat_ex_rot_base',
        )
        if any(name not in out for name in required):
            return None, None, None
        rot_q = torch.cat(
            (out['feat_x_base'], out['feat_y_base'], out['feat_z_base']),
            dim=-1,
        ).view(batch_size, -1)
        feat_ex_rot = out['feat_ex_rot_base'].view(batch_size, -1)
        return rot_q, feat_ex_rot[:, :2], feat_ex_rot[:, 2:]



    def update(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        loss_scale: float = 1.0,
        reset_gradients: bool = True,
        step_optimizer: bool = True,
        return_visualization: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        oracle_points, oracle_valid, oracle_slots = (
            self._select_oracle_prior_points(replay_sample)
        )
        relation_state = latest_replay_value(
            replay_sample['low_dim_state'], 2,
        ).float()[:, :3]
        tasks = replay_sample["tasks"]
        return_out = {}
        if oracle_valid is not None:
            if oracle_valid.ndim == 2:
                return_out['oracle_target_coverage'] = (
                    oracle_valid[:, 0].float().mean().item()
                )
                return_out['oracle_reference_coverage'] = (
                    oracle_valid[:, 1].float().mean().item()
                )
                return_out['oracle_prior_coverage'] = (
                    oracle_valid.all(dim=1).float().mean().item()
                )
            else:
                return_out['oracle_prior_coverage'] = (
                    oracle_valid.float().mean().item()
                )

        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            oracle_scene_point_count = None
            oracle_point_shape = (
                tuple(oracle_points.shape[1:])
                if oracle_points is not None else None
            )
            if (
                oracle_points is not None
                and self._transform_augmentation
                and backprop
            ):
                oracle_scene_point_count = pc.shape[1]
                flat_oracle_points = oracle_points.reshape(
                    oracle_points.shape[0], -1, 3,
                )
                pc = torch.cat(
                    (pc, flat_oracle_points.to(pc.device)), dim=1,
                )

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
                if oracle_scene_point_count is not None:
                    oracle_points = pc[:, oracle_scene_point_count:].reshape(
                        pc.shape[0], *oracle_point_shape,
                    )
                    pc = pc[:, :oracle_scene_point_count]

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)  
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            if oracle_points is not None:
                oracle_points = torch.stack([
                    mvt_utils.place_pc_in_cube(
                        scene_pc,
                        app_pc=instance_points.reshape(-1, 3).to(
                            device=scene_pc.device, dtype=scene_pc.dtype
                        ),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean
                        else self.scene_bounds,
                    )[0].reshape(oracle_point_shape)
                    for scene_pc, instance_points in zip(pc, oracle_points)
                ])

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            oracle_compute_base=(backprop and self.oracle_log_base_loss),
            **self._oracle_network_kwargs(
                oracle_points, oracle_valid, relation_state,
            ),
            language_goal=replay_sample["lang_goal"]  
        )
        
        if self.internal_object_slots_enabled:
            slot_stages = [out]
            if self.stage_two:
                slot_stages.append(out['mvt2'])
            slot_confidence = torch.stack(
                [stage['object_slot_confidence'] for stage in slot_stages]
            ).mean(dim=0)
            slot_valid = torch.stack(
                [stage['object_slot_valid'].float() for stage in slot_stages]
            ).mean(dim=0)
            return_out.update({
                'object_slot_target_confidence': (
                    slot_confidence[:, 0].mean().item()
                ),
                'object_slot_reference_confidence': (
                    slot_confidence[:, 1].mean().item()
                ),
                'object_slot_target_valid_rate': (
                    slot_valid[:, 0].mean().item()
                ),
                'object_slot_reference_valid_rate': (
                    slot_valid[:, 1].mean().item()
                ),
            })

        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )
        base_q_trans = self.get_base_q_trans(out, dims=(bs, nc, h, w))
        base_rot_q, base_grip_q, base_collision_q = self.get_base_q_rgc(
            out, bs,
        )


        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss_values = self._cross_entropy_loss(q_trans, action_trans)
            trans_loss = trans_loss_values.mean()
            valid_loss_values = (
                trans_loss_values
                if self.oracle_valid_only_loss else trans_loss_values.detach()
            )
            trans_loss_valid = (
                valid_oracle_translation_loss(
                    valid_loss_values,
                    oracle_valid,
                    distributed=self.oracle_valid_only_loss,
                )
                if oracle_valid is not None else None
            )
            optimized_trans_loss = choose_oracle_translation_loss(
                trans_loss,
                trans_loss_valid,
                self.oracle_valid_only_loss,
            )
            base_trans_loss_values = (
                self._cross_entropy_loss(base_q_trans, action_trans)
                if base_q_trans is not None else None
            )
            base_trans_loss = (
                base_trans_loss_values.mean()
                if base_trans_loss_values is not None else None
            )
            base_trans_loss_valid = (
                valid_oracle_translation_loss(
                    base_trans_loss_values,
                    oracle_valid,
                    distributed=self.oracle_valid_only_loss,
                )
                if oracle_valid is not None
                and base_trans_loss_values is not None else None
            )
            zero_loss = trans_loss.new_zeros(())
            rot_loss_x = rot_loss_y = rot_loss_z = zero_loss
            grip_loss = zero_loss
            collision_loss = zero_loss
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            base_rot_loss_x = base_rot_loss_y = base_rot_loss_z = None
            base_grip_loss = base_collision_loss = None
            if self.add_rgc_loss and base_rot_q is not None:
                with torch.no_grad():
                    base_rot_loss_x = self._cross_entropy_loss(
                        base_rot_q[
                            :,
                            0 * self._num_rotation_classes:
                            1 * self._num_rotation_classes,
                        ],
                        action_rot_x_one_hot.argmax(-1),
                    ).mean()
                    base_rot_loss_y = self._cross_entropy_loss(
                        base_rot_q[
                            :,
                            1 * self._num_rotation_classes:
                            2 * self._num_rotation_classes,
                        ],
                        action_rot_y_one_hot.argmax(-1),
                    ).mean()
                    base_rot_loss_z = self._cross_entropy_loss(
                        base_rot_q[
                            :,
                            2 * self._num_rotation_classes:
                            3 * self._num_rotation_classes,
                        ],
                        action_rot_z_one_hot.argmax(-1),
                    ).mean()
                    base_grip_loss = self._cross_entropy_loss(
                        base_grip_q,
                        action_grip_one_hot.argmax(-1),
                    ).mean()
                    base_collision_loss = self._cross_entropy_loss(
                        base_collision_q,
                        action_collision_one_hot.argmax(-1),
                    ).mean()

            action_total_loss = (
                optimized_trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
            total_loss = action_total_loss
            object_slot_losses = self._object_slot_auxiliary_losses(
                out, oracle_valid,
                role_present=(torch.stack((
                    replay_sample['oracle_target_present'].reshape(-1),
                    replay_sample['oracle_reference_present'].reshape(-1),
                ), dim=1) if 'oracle_role_present_known' in replay_sample else None),
                role_present_known=replay_sample.get('oracle_role_present_known'),
            )
            if object_slot_losses:
                total_loss = (
                    total_loss
                    + self.object_slot_mask_loss_weight
                    * object_slot_losses['mask']
                    + self.object_slot_null_loss_weight
                    * object_slot_losses['presence']
                    + self.object_slot_diversity_loss_weight
                    * object_slot_losses['diversity']
            )
            base_total_loss = None
            if base_trans_loss is not None:
                base_total_loss = choose_oracle_translation_loss(
                    base_trans_loss,
                    base_trans_loss_valid,
                    self.oracle_valid_only_loss,
                )
                if self.add_rgc_loss:
                    if base_rot_loss_x is None:
                        raise RuntimeError(
                            'Base R/G/C logits are missing while base loss '
                            'comparison is enabled.'
                        )
                    base_total_loss = (
                        base_total_loss
                        + base_rot_loss_x
                        + base_rot_loss_y
                        + base_rot_loss_z
                        + base_grip_loss
                        + base_collision_loss
                    )


            if reset_gradients:
                self.zero_grad()

            (total_loss * loss_scale).backward()
            if step_optimizer:
                self.optimizer_step()


            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            if object_slot_losses:
                loss_log['action_total_loss'] = action_total_loss.item()
                loss_log.update({
                    'object_slot_mask_loss': object_slot_losses['mask'].item(),
                    'object_slot_presence_loss': (
                        object_slot_losses['presence'].item()
                    ),
                    'object_slot_diversity_loss': (
                        object_slot_losses['diversity'].item()
                    ),
                })
            if base_trans_loss is not None:
                loss_log['trans_loss_base'] = base_trans_loss.item()
            if trans_loss_valid is not None:
                loss_log['trans_loss_valid'] = trans_loss_valid.item()
            if base_trans_loss_valid is not None:
                loss_log['trans_loss_base_valid'] = (
                    base_trans_loss_valid.item()
                )
            if base_total_loss is not None:
                total_loss_gain = base_total_loss - action_total_loss.detach()
                loss_log['total_loss_base'] = base_total_loss.item()
                loss_log['total_loss_gain'] = total_loss_gain.item()
                loss_log['total_loss_gain_pct'] = (
                    100.0 * total_loss_gain
                    / base_total_loss.abs().clamp_min(1e-12)
                ).item()
            if base_rot_loss_x is not None:
                loss_log.update({
                    'rot_loss_x_base': base_rot_loss_x.item(),
                    'rot_loss_y_base': base_rot_loss_y.item(),
                    'rot_loss_z_base': base_rot_loss_z.item(),
                    'grip_loss_base': base_grip_loss.item(),
                    'collision_loss_base': base_collision_loss.item(),
                })
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)
            if return_visualization:
                return_out['train_visualization'] = (
                    build_training_visualization_payload(
                        out,
                        action_trans,
                        num_views=nc,
                        height=h,
                        width=w,
                        stage_two=self.stage_two,
                    )
                )


        return return_out



    def update_gembench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
    ) -> dict:
        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"]  # (b, 8)  
        

        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3) 
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4) 

        action_grip = action_gripper_pose[:, -1].int()   # (b,)
        return_out = {}

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )
            import open3d as o3d
            def vis_pcd(pc, rgb,save_path):

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)  
                pcd.colors = o3d.utility.Vector3dVector(rgb) 
                o3d.io.write_point_cloud(save_path, pcd)
                # o3d.visualization.draw_geometries([pcd])
            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
            
            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )

        loss_log = {}
        if backprop:
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()  
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward()
            self._optimizer.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out


    @torch.no_grad()
    def _decode_action_waypoint(
        self, output, rev_trans, dyn_cam_info, *, use_base,
    ):
        attribution_output = dict(output)
        if self.stage_two:
            stage_output = dict(output['mvt2'])
            if use_base:
                stage_output['trans'] = stage_output.get(
                    'trans_base', stage_output['trans'])
            attribution_output['mvt2'] = stage_output
            first_stage = False
        else:
            if use_base:
                attribution_output['trans'] = output.get(
                    'trans_base', output['trans'])
            first_stage = True
        waypoint_local = self._net_mod.get_wpt(
            attribution_output, first_stage, dyn_cam_info, None)
        return torch.cat([
            reverse(value).unsqueeze(0)
            for value, reverse in zip(waypoint_local, rev_trans)
        ])

    @torch.no_grad()
    def _heatmap_action_anchor_replay_elements(
        self, output, observation, rev_trans, dyn_cam_info, final_waypoint,
    ):
        """Attribute base and executed translation waypoints without redefining GT."""
        required = (
            'oracle_target_candidate_points',
            'oracle_target_candidate_valid',
            'oracle_target_candidate_phase_indices',
            'oracle_target_current_candidate_index',
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(
                'Heatmap action-anchor attribution requires simulator candidates: '
                + ', '.join(missing)
            )

        candidates = latest_replay_value(
            observation['oracle_target_candidate_points'], 4).float()
        valid = latest_replay_value(
            observation['oracle_target_candidate_valid'], 2).bool()
        phase_indices = latest_replay_value(
            observation['oracle_target_candidate_phase_indices'], 2).long()
        current_indices = latest_replay_value(
            observation['oracle_target_current_candidate_index'], 1).long()
        eligible_targets = active_semantic_target_mask(valid, phase_indices)
        oracle_current_index = int(current_indices[0].item())
        current_index = oracle_current_index
        if self.bridgevla_aligned_objects and self._bridgevla_target_lock >= 0:
            current_index = self._bridgevla_target_lock

        def attribute(prefix, waypoint):
            selected, distance, confidence = (
                select_object_candidate_from_waypoint(
                    waypoint, candidates, valid))
            eligible, eligible_distance, eligible_confidence = (
                select_object_candidate_from_waypoint(
                    waypoint, candidates, eligible_targets))
            index = int(selected[0].item())
            eligible_index = int(eligible[0].item())
            phase = (
                int(phase_indices[0, index].item()) if index >= 0 else -1)
            values = {
                f'{prefix}_candidate_index': np.asarray(index, dtype=np.int64),
                f'{prefix}_candidate_phase_index': np.asarray(
                    phase, dtype=np.int64),
                f'{prefix}_matches_current_target': np.asarray(
                    index >= 0 and index == current_index, dtype=np.bool_),
                f'{prefix}_distance_m': np.asarray(
                    float(distance[0].item()), dtype=np.float32),
                f'{prefix}_confidence': np.asarray(
                    float(confidence[0].item()), dtype=np.float32),
                f'{prefix}_waypoint': waypoint[0].detach().cpu().numpy(),
                f'{prefix}_eligible_target_candidate_index': np.asarray(
                    eligible_index, dtype=np.int64),
                f'{prefix}_eligible_target_distance_m': np.asarray(
                    float(eligible_distance[0].item()), dtype=np.float32),
                f'{prefix}_eligible_target_confidence': np.asarray(
                    float(eligible_confidence[0].item()), dtype=np.float32),
            }
            if (
                'oracle_reference_object_points' in observation
                and 'oracle_reference_object_valid' in observation
            ):
                reference = latest_replay_value(
                    observation['oracle_reference_object_points'], 3).float()
                reference_valid = latest_replay_value(
                    observation['oracle_reference_object_valid'], 1).bool()
                near_reference, reference_distance, reference_confidence = (
                    select_object_candidate_from_waypoint(
                        waypoint, reference[:, None], reference_valid[:, None]))
                values.update({
                    f'{prefix}_near_current_reference': np.asarray(
                        int(near_reference[0].item()) == 0, dtype=np.bool_),
                    f'{prefix}_reference_distance_m': np.asarray(
                        float(reference_distance[0].item()), dtype=np.float32),
                    f'{prefix}_reference_confidence': np.asarray(
                        float(reference_confidence[0].item()), dtype=np.float32),
                })
            return values, index

        base_waypoint = self._decode_action_waypoint(
            output, rev_trans, dyn_cam_info, use_base=True)
        base_stage_output = output['mvt2'] if self.stage_two else output
        elements = {
            'heatmap_action_anchor_current_target_candidate_index': np.asarray(
                current_index, dtype=np.int64),
            'heatmap_action_anchor_oracle_target_candidate_index': np.asarray(
                oracle_current_index, dtype=np.int64),
            'heatmap_action_anchor_base_available': np.asarray(
                'trans_base' in base_stage_output, dtype=np.bool_),
            'heatmap_action_anchor_eligible_target_mask': (
                eligible_targets[0].detach().cpu().numpy()),
        }
        base_values, _ = attribute(
            'heatmap_action_anchor_base', base_waypoint)
        final_values, final_index = attribute(
            'heatmap_action_anchor_final', final_waypoint)
        elements.update(base_values)
        elements.update(final_values)
        if final_index >= 0:
            selected_points = candidates[0, final_index].detach().cpu().numpy()
        else:
            selected_points = np.zeros(
                tuple(candidates.shape[2:]), dtype=np.float32)
        elements['heatmap_action_anchor_object_points'] = selected_points
        elements['heatmap_action_anchor_object_valid'] = np.asarray(
            final_index >= 0, dtype=np.bool_)
        return elements

    @torch.no_grad()
    def _bridgevla_aligned_relation(
        self, output, observation, relation_state, oracle_points, oracle_valid,
        candidate_points_local, candidate_reference_points_local,
        rev_trans, dyn_cam_info,
    ):
        """Select residual T/R geometry from the base BridgeVLA action intent."""
        if oracle_points is None or oracle_points.ndim != 4:
            raise ValueError(
                'BridgeVLA-aligned objects require relation points [B,2,P,3]')
        if oracle_points.shape[0] != 1:
            raise ValueError(
                'BridgeVLA-aligned objects currently require evaluation batch size 1')
        base_stage = output['mvt2'] if self.stage_two else output
        if 'trans_base' not in base_stage:
            raise ValueError(
                'BridgeVLA-aligned objects require trans_base before residual adaptation')

        candidates = latest_replay_value(
            observation['oracle_target_candidate_points'], 4).float()
        valid = latest_replay_value(
            observation['oracle_target_candidate_valid'], 2).bool()
        phases = latest_replay_value(
            observation['oracle_target_candidate_phase_indices'], 2).long()
        base_waypoint = self._decode_action_waypoint(
            output, rev_trans, dyn_cam_info, use_base=True)
        proposed, distance, confidence = select_object_candidate_from_waypoint(
            base_waypoint, candidates, valid)
        proposed_index = int(proposed[0].item())

        gripper_open = bool(relation_state[0, 0].item() > 0.5)
        released = (
            self._bridgevla_last_gripper_open is False and gripper_open)
        if released:
            self._bridgevla_target_lock = -1
        if self._bridgevla_target_lock < 0 and proposed_index >= 0:
            self._bridgevla_target_lock = proposed_index
        self._bridgevla_last_gripper_open = gripper_open

        locked_index = self._bridgevla_target_lock
        lock_usable = (
            0 <= locked_index < valid.shape[1]
            and bool(valid[0, locked_index].item())
        )
        aligned_points = oracle_points
        aligned_valid = oracle_valid
        reference_source = 0  # 0=current task Reference, 1=phase-paired Reference
        if lock_usable:
            aligned_points = oracle_points.clone()
            aligned_valid = oracle_valid.clone()
            aligned_points[0, 0] = candidate_points_local[0, locked_index]
            aligned_valid[0, 0] = True

            paired_reference_valid = latest_replay_value(
                observation['oracle_target_candidate_reference_valid'], 2,
            ).bool()
            if bool(paired_reference_valid[0, locked_index].item()):
                aligned_points[0, 1] = (
                    candidate_reference_points_local[0, locked_index])
                aligned_valid[0, 1] = True
                reference_source = 1

        phase_index = (
            int(phases[0, locked_index].item()) if lock_usable else -1)
        elements = {
            'bridgevla_aligned_target_proposed_index': np.asarray(
                proposed_index, dtype=np.int64),
            'bridgevla_aligned_target_locked_index': np.asarray(
                locked_index if lock_usable else -1, dtype=np.int64),
            'bridgevla_aligned_target_phase_index': np.asarray(
                phase_index, dtype=np.int64),
            'bridgevla_aligned_target_distance_m': np.asarray(
                float(distance[0].item()), dtype=np.float32),
            'bridgevla_aligned_target_confidence': np.asarray(
                float(confidence[0].item()), dtype=np.float32),
            'bridgevla_aligned_target_used': np.asarray(
                lock_usable, dtype=np.bool_),
            'bridgevla_aligned_reference_source': np.asarray(
                reference_source, dtype=np.int64),
            'bridgevla_aligned_gripper_open': np.asarray(
                gripper_open, dtype=np.bool_),
            'bridgevla_aligned_released_lock': np.asarray(
                released, dtype=np.bool_),
        }
        return aligned_points, aligned_valid, elements

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=False,
        visualize=False, visualize_save_dir="", return_gembench_action=False,
    ) -> ActResult:
        oracle_points, oracle_valid, oracle_slots = (
            self._select_oracle_prior_points(observation, allow_missing=True)
        )
        relation_state = latest_replay_value(
            observation['low_dim_state'], 2,
        ).float()[:, :3]
        language_goal =observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_ori = pc[0].clone()
        img_feat_ori=img_feat[0].clone()
        aligned_candidate_points_world = None
        aligned_candidate_reference_world = None
        aligned_candidate_points_local = None
        aligned_candidate_reference_local = None
        if self.bridgevla_aligned_objects:
            required = (
                'oracle_target_candidate_points',
                'oracle_target_candidate_valid',
                'oracle_target_candidate_phase_indices',
                'oracle_target_candidate_reference_points',
                'oracle_target_candidate_reference_valid',
            )
            missing = [key for key in required if key not in observation]
            if missing:
                raise KeyError(
                    'BridgeVLA-aligned residual objects require simulator '
                    'candidates: ' + ', '.join(missing))
            aligned_candidate_points_world = latest_replay_value(
                observation['oracle_target_candidate_points'], 4).float()
            aligned_candidate_reference_world = latest_replay_value(
                observation['oracle_target_candidate_reference_points'], 4,
            ).float()
            aligned_candidate_points_local = []
            aligned_candidate_reference_local = []
        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        oracle_points_local = [] if oracle_points is not None else None
        oracle_point_shape = (
            tuple(oracle_points.shape[1:])
            if oracle_points is not None else None
        )
        for batch_index, _pc in enumerate(pc):
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
            if oracle_points_local is not None:
                oracle_points_local.append(
                    mvt_utils.place_pc_in_cube(
                        _pc,
                        app_pc=oracle_points[batch_index].reshape(-1, 3).to(
                            device=_pc.device, dtype=_pc.dtype
                        ),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean
                        else self.scene_bounds,
                    )[0].reshape(oracle_point_shape)
                )
            if aligned_candidate_points_local is not None:
                candidate_shape = tuple(
                    aligned_candidate_points_world.shape[1:])
                aligned_candidate_points_local.append(
                    mvt_utils.place_pc_in_cube(
                        _pc,
                        app_pc=aligned_candidate_points_world[
                            batch_index].reshape(-1, 3).to(
                                device=_pc.device, dtype=_pc.dtype),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean
                        else self.scene_bounds,
                    )[0].reshape(candidate_shape)
                )
                aligned_candidate_reference_local.append(
                    mvt_utils.place_pc_in_cube(
                        _pc,
                        app_pc=aligned_candidate_reference_world[
                            batch_index].reshape(-1, 3).to(
                                device=_pc.device, dtype=_pc.dtype),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean
                        else self.scene_bounds,
                    )[0].reshape(candidate_shape)
                )
        pc = pc_new
        if oracle_points_local is not None:
            oracle_points = torch.stack(oracle_points_local)
        if aligned_candidate_points_local is not None:
            aligned_candidate_points_local = torch.stack(
                aligned_candidate_points_local)
            aligned_candidate_reference_local = torch.stack(
                aligned_candidate_reference_local)

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,  # no img augmentation while acting
            oracle_compute_base=(
                self.heatmap_action_anchor
                or self.bridgevla_aligned_objects
            ),
            **self._oracle_network_kwargs(
                oracle_points, oracle_valid, relation_state,
            ),
            language_goal=language_goal,
        )
        bridgevla_alignment_elements = {}
        if self.bridgevla_aligned_objects:
            aligned_points, aligned_valid, bridgevla_alignment_elements = (
                self._bridgevla_aligned_relation(
                    out, observation, relation_state, oracle_points, oracle_valid,
                    aligned_candidate_points_local,
                    aligned_candidate_reference_local,
                    rev_trans, dyn_cam_info,
                )
            )
            if bool(bridgevla_alignment_elements[
                'bridgevla_aligned_target_used']):
                out = self._network(
                    pc=pc,
                    img_feat=img_feat,
                    img_aug=0,
                    oracle_compute_base=True,
                    **self._oracle_network_kwargs(
                        aligned_points, aligned_valid, relation_state,
                    ),
                    language_goal=language_goal,
                )
        if visualize:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )            
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )
        heatmap_action_anchor_elements = {}
        if self.heatmap_action_anchor:
            heatmap_action_anchor_elements = (
                self._heatmap_action_anchor_replay_elements(
                    out, observation, rev_trans, dyn_cam_info,
                    final_waypoint=pred_wpt)
            )
            diagnostic_step = self._heatmap_action_anchor_step
            self._heatmap_action_anchor_step += 1
            heatmap_action_anchor_elements[
                'heatmap_action_anchor_policy_step'] = np.asarray(
                    diagnostic_step, dtype=np.int64)
            print(
                '[HeatmapActionAnchor] '
                f'step={diagnostic_step} '
                f'current_target={int(heatmap_action_anchor_elements["heatmap_action_anchor_current_target_candidate_index"])} '
                f'oracle_target={int(heatmap_action_anchor_elements["heatmap_action_anchor_oracle_target_candidate_index"])} '
                f'base={int(heatmap_action_anchor_elements["heatmap_action_anchor_base_candidate_index"])} '
                f'base_available={bool(heatmap_action_anchor_elements["heatmap_action_anchor_base_available"])} '
                f'base_phase={int(heatmap_action_anchor_elements["heatmap_action_anchor_base_candidate_phase_index"])} '
                f'base_eligible={int(heatmap_action_anchor_elements["heatmap_action_anchor_base_eligible_target_candidate_index"])} '
                f'base_reference_distance_m={float(heatmap_action_anchor_elements["heatmap_action_anchor_base_reference_distance_m"]):.4f} '
                f'final={int(heatmap_action_anchor_elements["heatmap_action_anchor_final_candidate_index"])} '
                f'final_phase={int(heatmap_action_anchor_elements["heatmap_action_anchor_final_candidate_phase_index"])} '
                f'final_eligible={int(heatmap_action_anchor_elements["heatmap_action_anchor_final_eligible_target_candidate_index"])} '
                f'final_matches_target={bool(heatmap_action_anchor_elements["heatmap_action_anchor_final_matches_current_target"])} '
                f'final_near_reference={bool(heatmap_action_anchor_elements["heatmap_action_anchor_final_near_current_reference"])} '
                f'final_reference_distance_m={float(heatmap_action_anchor_elements["heatmap_action_anchor_final_reference_distance_m"]):.4f} '
                f'final_distance_m={float(heatmap_action_anchor_elements["heatmap_action_anchor_final_distance_m"]):.4f}',
                flush=True,
            )
        if self.bridgevla_aligned_objects:
            heatmap_action_anchor_elements.update(bridgevla_alignment_elements)
            print(
                '[BridgeVLAAlignedObjects] '
                f'step={self._heatmap_action_anchor_step - 1} '
                f'proposed={int(bridgevla_alignment_elements["bridgevla_aligned_target_proposed_index"])} '
                f'locked={int(bridgevla_alignment_elements["bridgevla_aligned_target_locked_index"])} '
                f'phase={int(bridgevla_alignment_elements["bridgevla_aligned_target_phase_index"])} '
                f'used={bool(bridgevla_alignment_elements["bridgevla_aligned_target_used"])} '
                f'reference_source={int(bridgevla_alignment_elements["bridgevla_aligned_reference_source"])}',
                flush=True,
            )
        if visualize:
            print("Visualizing")
            save_dir=visualize_save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir=os.path.join(save_dir,f"step{str(step)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            stage_outputs = [
                ('mvt1', out, out['mvt1_ori_img'][0, :, 3:6]),
            ]
            if self.stage_two and 'mvt2' in out:
                stage_outputs.append(
                    ('mvt2', out['mvt2'], out['mvt2_ori_img'][0, :, 3:6])
                )
            for stage_name, stage_out, stage_img in stage_outputs:
                stage_dir = os.path.join(save_dir, stage_name)
                final = translation_heatmap_probabilities(
                    stage_out['trans'][0]
                )
                visualize_images(stage_img, final, save_dir=stage_dir)
                if 'oracle_instance_prior' in stage_out:
                    if 'oracle_target_prior' in stage_out:
                        save_heatmap_views(
                            stage_out['oracle_target_prior'][0],
                            stage_dir,
                            'o2_target_prior',
                            stage_img,
                        )
                    if 'oracle_reference_prior' in stage_out:
                        save_heatmap_views(
                            stage_out['oracle_reference_prior'][0],
                            stage_dir,
                            'o2_reference_prior',
                            stage_img,
                        )
                    save_heatmap_views(
                        stage_out['oracle_instance_prior'][0],
                        stage_dir,
                        'o2_prior',
                        stage_img,
                    )
                    if 'oracle_relation_anchor' in stage_out:
                        save_heatmap_views(
                            stage_out['oracle_relation_anchor'][0],
                            stage_dir,
                            'o2_relation_anchor',
                            stage_img,
                        )
                    save_heatmap_views(
                        final, stage_dir, 'o2_adapted', stage_img,
                    )
                elif self.object_prior_enabled:
                    with open(
                        os.path.join(stage_dir, 'o2_unavailable.txt'),
                        'w',
                        encoding='utf-8',
                    ) as stream:
                        stream.write(
                            'Oracle prior unavailable; this step used base '
                            'BridgeVLA translation logits.\\n'
                        )
            save_point_cloud_with_color(os.path.join(save_dir,"point_cloud.ply"), pc_ori.cpu().numpy(), img_feat_ori.cpu().numpy(), pred_wpt[0].cpu().numpy())
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
                # [1.0],  # debug!!!!!!
            )
        )

        if return_gembench_action:
            continuous_action = np.concatenate(
                    [
                        pred_wpt[0].cpu().numpy(),
                        pred_rot_quat[0],
                        pred_grip[0].cpu().numpy(),
                    ], -1
                )
            return continuous_action
        else:
            return ActResult(
                continuous_action,
                replay_elements=heatmap_action_anchor_elements,
            )



    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll


    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans



    def reset(self):
        self._heatmap_action_anchor_step = 0
        self._bridgevla_target_lock = -1
        self._bridgevla_last_gripper_open = None

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
