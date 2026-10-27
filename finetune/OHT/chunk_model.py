"""Prediction-only chunk context, with optional joint key-action supervision."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .action_modes import action_options, validate_action
from .chunk_config import chunk_settings


def rotation_6d(value):
    first = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    second = F.normalize(value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def quaternion_matrix(quaternion):
    q = F.normalize(quaternion, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class ChunkDecoder(nn.Module):
    def __init__(self, feature_dim, settings):
        super().__init__()
        self.settings = settings
        dim = settings['hidden_dim']
        self.visual = nn.Linear(feature_dim, dim)
        self.state = nn.Linear(14, dim)  # current TCP7 + q6 + measured opening1
        self.goal = nn.Linear(8, dim)
        self.scale_embedding = nn.Parameter(torch.zeros(2, dim))
        self.queries = nn.Parameter(torch.randn(settings['horizon'], dim) * .02)
        layer = nn.TransformerDecoderLayer(dim, settings['heads'], dim * 4,
                                           dropout=settings['dropout'], batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, settings['layers'])
        self.head = nn.Linear(dim, 10)
        nn.init.normal_(self.head.weight, std=.001)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[3:9] = torch.tensor([1., 0., 0., 0., 1., 0.])

    def forward(self, coarse, fine, state, goal):
        dtype = self.visual.weight.dtype
        goal_condition = self.goal_condition(state, goal)
        memory = torch.cat((self.visual(coarse.to(dtype)) + self.scale_embedding[0],
                            self.visual(fine.to(dtype)) + self.scale_embedding[1],
                            self.state(state.to(dtype))[:, None], self.goal(goal_condition.to(dtype))[:, None]), 1)
        query = self.queries[None].expand(state.shape[0], -1, -1)
        raw = self.head(self.transformer(query, memory))
        position = state[:, None, :3] + raw[..., :3] * self.settings['position_scale']
        return position, rotation_6d(raw[..., 3:9]), raw[..., 9]

    def goal_condition(self, state, goal):
        if self.settings['goal_conditioning'] == 'absolute':
            return goal
        # Only the decoder's condition changes. State and returned EEF actions
        # remain in world coordinates; goal is the detached predicted key pose.
        current = F.normalize(state[:, 3:7], dim=-1)
        target = F.normalize(goal[:, 3:7], dim=-1)
        local_position = torch.einsum('bij,bj->bi', quaternion_matrix(current).transpose(-1, -2),
                                      goal[:, :3] - state[:, :3]) / self.settings['position_scale']
        xyz, w = current[:, :3], current[:, 3:4]
        target_xyz, target_w = target[:, :3], target[:, 3:4]
        relative = torch.cat((w * target_xyz - target_w * xyz - torch.linalg.cross(xyz, target_xyz),
                              w * target_w + (xyz * target_xyz).sum(-1, keepdim=True)), dim=-1)
        relative = F.normalize(relative, dim=-1)
        # q and -q encode the same rotation, including exactly 180 degrees.
        pivot = relative.gather(-1, relative.abs().argmax(-1, keepdim=True))
        relative = torch.where(pivot < 0, -relative, relative)
        return torch.cat((local_position, relative, goal[:, 7:8]), dim=-1)


def chunk_loss(prediction, target, valid, settings, *, current_position=None):
    position, rotation, grip = prediction
    target, valid = target[:, :position.shape[1]], valid[:, :position.shape[1]]
    if target.shape != (*position.shape[:2], 8) or valid.shape != position.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Chunk supervision requires [B,K,8] and bool [B,K]')
    if not valid.any() or not torch.isfinite(target[valid]).all():
        raise ValueError('Chunk batch has no valid finite supervision')
    p = F.smooth_l1_loss(position[valid] / settings['position_scale'],
                         target[..., :3][valid] / settings['position_scale'])
    r = F.mse_loss(rotation[valid], quaternion_matrix(target[..., 3:7][valid]))
    g = F.binary_cross_entropy_with_logits(grip[valid], target[..., 7][valid])
    weighted_p = settings['position_weight'] * p
    displacement = position[valid].sum() * 0.
    displacement_error = stationary_displacement_error = 0.
    if settings['displacement_weight'] > 0 and current_position is None:
        raise ValueError('Displacement supervision requires the measured current TCP position')
    if current_position is not None:
        if current_position.shape != (position.shape[0], 3) or not torch.isfinite(current_position).all():
            raise ValueError('Current TCP position must be finite [B,3]')
        # Anchor the first tick to the measured state. Later ticks compare only
        # consecutive valid labels, never bridging padding or missing frames.
        paired = torch.cat((valid[:, :1], valid[:, 1:] & valid[:, :-1]), dim=1)
        if paired.any():
            previous_predicted = torch.cat((current_position[:, None], position[:, :-1]), dim=1)
            previous_target = torch.cat((current_position[:, None], target[:, :-1, :3]), dim=1)
            predicted_delta = position[paired] - previous_predicted[paired]
            target_delta = target[..., :3][paired] - previous_target[paired]
            scale = settings['displacement_scale']
            displacement = F.smooth_l1_loss(predicted_delta / scale, target_delta / scale)
            with torch.no_grad():
                displacement_error = torch.linalg.vector_norm(predicted_delta - target_delta, dim=-1).mean().item()
                stationary_displacement_error = torch.linalg.vector_norm(target_delta, dim=-1).mean().item()
    weighted_d = settings['displacement_weight'] * displacement
    return weighted_p + weighted_d + settings['rotation_weight'] * r + settings['gripper_weight'] * g, dict(
        chunk_position_loss=p.detach().item(), chunk_position_weighted_loss=weighted_p.detach().item(),
        chunk_displacement_loss=displacement.detach().item(),
        chunk_displacement_weighted_loss=weighted_d.detach().item(),
        chunk_displacement_error_m=displacement_error,
        chunk_stationary_displacement_error_m=stationary_displacement_error,
        chunk_rotation_loss=r.detach().item(),
        chunk_gripper_loss=g.detach().item(), chunk_valid_actions=int(valid.sum().item()))


class ChunkNetwork(nn.Module):
    def __init__(self, encoder, decoder, stage='frozen_key'):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.stage = stage
        if stage == 'frozen_key':
            self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # never activate GT crop/teacher-forced rotation routing
        return self

    def forward(self, *args, context_builder=None, key_loss_builder=None):
        if context_builder is None:
            return self.decoder(*args)
        # All trainable forwards are inside the outer DDP call, including both
        # uses of the shared encoder. The GT branch never supplies decoder inputs.
        key_loss, metrics = key_loss_builder() if key_loss_builder else (None, {})
        prediction = self.decoder(*context_builder())
        return prediction, key_loss, metrics


class ChunkAgent:
    is_chunk_agent = True

    def __init__(self, base, settings, contract, config, device, distributed=False, training=True):
        self.base, self.settings, self.contract = base, settings, contract
        self.oht_action_config = action_options()
        self._device = device
        base._net_mod.mvt1.oht_return_chunk_tokens = True
        dim = base._net_mod.mvt1.vlm_dim
        self._net_mod = ChunkNetwork(base._net_mod, ChunkDecoder(dim, settings), settings['stage']).to(device)
        self._net_mod.train(training)
        self._network = self._net_mod
        self.distributed = distributed
        if distributed:
            self._network = torch.nn.parallel.DistributedDataParallel(
                self._net_mod, device_ids=[device.index] if device.type == 'cuda' else None,
                find_unused_parameters=settings['stage'] == 'joint')
        optimizer = config['agent'].get('optimizer_type', 'adamw').lower()
        optimizer_class = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(optimizer)
        if optimizer_class is None:
            raise ValueError('Chunk optimizer must be Adam or AdamW')
        self._optimizer = optimizer_class((p for p in self._net_mod.parameters() if p.requires_grad), lr=config['learning_rate'],
                                         weight_decay=config['agent'].get('lambda_weight_l2', 0.))

    def reset(self):
        self.base.reset()

    def _context(self, sample, visualization_callback=None, with_grad=False):
        from .runtime.predicted_wrapper import policy_observation
        observation = policy_observation(sample, self.contract['data_config']['cameras'])
        for field, shape in (('tcp_pose', (7,)), ('joint_positions', (6,)), ('gripper_open01', ())):
            value = np.asarray(sample[field], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid current chunk proprioception: {field}')
        state = np.r_[sample['tcp_pose'], sample['joint_positions'], sample['gripper_open01']].astype(np.float32)
        norm = np.linalg.norm(state[3:7])
        if norm < 1e-8 or not 0 <= state[-1] <= 1:
            raise ValueError('Invalid measured TCP quaternion or gripper opening')
        state[3:7] /= norm
        if state[6] < 0:
            state[3:7] *= -1
        observation = {name: torch.as_tensor(value, device=self._device)[None, None]
                       for name, value in observation.items()}
        observation['language_goal'] = [[[sample['goal']]]]
        captured = []
        handle = self.base._net_mod.register_forward_hook(lambda module, args, output: captured.append(output))
        try:
            self.base._net_mod.eval()
            with torch.set_grad_enabled(with_grad):
                goal = self.base.act(0, observation, deterministic=True, return_gembench_action=True,
                                     **({'context_with_grad': True} if with_grad else {}),
                                     visualization_callback=visualization_callback)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Chunk encoder must make exactly one prediction-only backbone forward')
        output = captured[0]
        coarse = output['oht_chunk_tokens']
        fine = output.get('mvt2', output)['oht_chunk_tokens']
        if not with_grad:
            coarse, fine = coarse.detach(), fine.detach()
        goal = torch.as_tensor(validate_action(goal, self.oht_action_config), device=self._device)[None]
        return coarse, fine, torch.as_tensor(state, device=self._device)[None], goal

    def _key_loss(self, batch):
        valid = batch['key_valid'][:, 0]
        if valid.dtype != torch.bool or valid.ndim != 1:
            raise ValueError('Joint key supervision requires bool [B,1] key_valid')
        count = int(valid.sum().item())
        names = ('trans_loss', 'rot_loss_x', 'rot_loss_y', 'rot_loss_z', 'grip_loss', 'collision_loss')
        if count:
            result = self._supervised_key_loss(batch, valid)
            loss = result.pop('loss')
        else:
            loss, result = torch.zeros((), device=self._device), dict.fromkeys(names, 0.)
        # One reduction supplies both globally valid-key normalized gradients
        # and matching logged means, even when a rank contains only tail frames.
        stats = torch.tensor([count, count * loss.detach().item(),
                              *(count * result[name] for name in names)], device=self._device, dtype=torch.float64)
        world = 1
        if self.distributed:
            torch.distributed.all_reduce(stats)
            world = torch.distributed.get_world_size()
        if not torch.isfinite(stats).all():
            raise ValueError('Nonfinite joint key loss; refusing optimizer update on all ranks')
        total_count, total_loss, *components = stats.tolist()
        denominator = max(total_count, 1.)
        return loss * (count * world / denominator), dict(
            key_total_loss=total_loss / denominator, key_valid_samples=int(total_count), key_local_valid_samples=count,
            **{'key_' + name: value / denominator for name, value in zip(names, components)})

    def _supervised_key_loss(self, batch, valid):
        count = int(valid.sum().item())
        indices = valid.nonzero(as_tuple=True)[0].tolist()
        from .runtime.predicted_wrapper import observation_fields
        observation_names = observation_fields(self.contract['data_config']['cameras'])
        sample = {name: batch[name][valid] for name in observation_names}
        key = batch['key_action'][valid]
        if not torch.isfinite(key).all():
            raise ValueError('Joint key supervision contains nonfinite valid labels')
        sample.update(gripper_pose=key[..., :7],
                      rot_grip_action_indicies=torch.zeros(count, 1, 4, dtype=torch.int32, device=self._device),
                      ignore_collisions=torch.zeros(count, 1, 1, dtype=torch.int32, device=self._device),
                      lang_goal=[batch['lang_goal'][i] for i in indices], tasks=[batch['tasks'][i] for i in indices])
        sample['rot_grip_action_indicies'][..., -1] = key[..., 7].int()
        encoder = self._net_mod.encoder
        encoder.train()
        # Frozen VLM modules stay in eval. Hold BatchNorm statistics fixed so
        # singleton valid-key batches and the prediction branch share one state.
        for module in encoder.modules():
            parameters = list(module.parameters())
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or (parameters and not any(p.requires_grad for p in parameters)):
                module.eval()
        try:
            return self.base.update(sample, return_loss=True)
        finally:
            encoder.eval()

    def update(self, batch, loss_scale=1., reset_gradients=True, step_optimizer=True, return_visualization=False):
        payload = []
        batch_size = batch['tcp_pose'].shape[0]
        joint = self.settings['stage'] == 'joint'
        def context_builder():
            contexts = []
            # Match the deployed single-observation prediction/crop path.
            for i in range(batch_size):
                sample = {name: value[i, 0].detach().cpu().numpy() for name, value in batch.items()
                          if torch.is_tensor(value) and name not in ('oracle_role_present_known',)}
                goal = batch['lang_goal'][i]
                while isinstance(goal, (list, tuple)):
                    goal = goal[0]
                sample['goal'] = goal
                contexts.append(self._context(sample, payload.append if return_visualization and i == 0 else None,
                                              with_grad=joint))
            return tuple(torch.cat([row[j] for row in contexts], 0) for j in range(4))
        if reset_gradients:
            self._optimizer.zero_grad(set_to_none=True)
        prediction, key_loss, key_metrics = self._network(
            context_builder=context_builder, key_loss_builder=(lambda: self._key_loss(batch)) if joint else None)
        loss, metrics = chunk_loss(prediction, batch['action_chunk'][:, 0], batch['action_valid'][:, 0], self.settings,
                                   current_position=batch['tcp_pose'][:, 0, :3])
        with torch.no_grad():
            valid = batch['action_valid'][:, 0, :self.settings['horizon']]
            target = batch['action_chunk'][:, 0, :self.settings['horizon'], :3]
            error = torch.linalg.vector_norm(prediction[0] - target, dim=-1)
            stationary = torch.linalg.vector_norm(batch['tcp_pose'][:, 0, None, :3] - target, dim=-1)
            metrics.update(chunk_position_error_m=error[valid].mean().item(),
                           chunk_stationary_error_m=stationary[valid].mean().item())
        metrics.update(chunk_total_loss=loss.detach().item(), **key_metrics)
        if key_loss is not None:
            loss = loss + self.settings['key_loss_weight'] * key_loss
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite chunk loss; refusing optimizer update')
        (loss * loss_scale).backward()
        if step_optimizer:
            self._optimizer.step()
        reported_loss = metrics['chunk_total_loss'] + self.settings['key_loss_weight'] * key_metrics.get('key_total_loss', 0.)
        result = dict(total_loss=reported_loss, **metrics)
        if return_visualization:
            result['train_visualization'] = dict(chunk=True, policy=payload[0] if payload else None,
                predicted=prediction[0][0].detach().float().cpu().numpy(),
                target=batch['action_chunk'][0, 0, :self.settings['horizon'], :3].detach().float().cpu().numpy(),
                valid=batch['action_valid'][0, 0, :self.settings['horizon']].cpu().numpy(),
                current=batch['tcp_pose'][0, 0, :3].detach().float().cpu().numpy(),
                sample_id=batch.get('sample_ids', [''])[0], action_dt=self.contract['chunk']['action_dt'])
        return result

    def predict(self, observation, goal, visualization_callback=None):
        sample = dict(observation, goal=goal)
        with torch.no_grad():
            context = self._context(sample, visualization_callback)
            position, rotation, grip = self._net_mod(*context)
        matrix = rotation[0].detach().float().cpu().numpy()
        if not np.allclose(np.linalg.det(matrix), 1., atol=1e-4):
            raise ValueError('Chunk decoder produced a degenerate rotation')
        from scipy.spatial.transform import Rotation
        actions = np.c_[position[0].detach().float().cpu().numpy(), Rotation.from_matrix(matrix).as_quat(),
                        (grip[0] >= 0).cpu().numpy()].astype(np.float32)
        actions = np.stack([validate_action(action, self.oht_action_config) for action in actions])
        return dict(key_action=context[3][0].cpu().numpy(), actions=actions)


def build_chunk_agent(config, contract, device, pretrain_path=None, training=True, distributed=False):
    from .model import build_agent
    settings = chunk_settings(config, contract)
    base_config = copy.deepcopy(config)
    base_config.pop('chunk_policy')
    base = build_agent(base_config, contract, device, pretrain_path,
                       training=training and settings['stage'] == 'joint', distributed=False)
    return ChunkAgent(base, settings, contract, config, device, distributed, training)


def load_chunk_weights(agent, checkpoint, initialize=False):
    from .model import load_weights
    saved = chunk_settings(checkpoint.get('config', {}))
    if not saved['enabled']:
        if not initialize:
            raise ValueError('Single-key checkpoint only supports --init-checkpoint for a new chunk head')
        previous = checkpoint.get('oht_contract', {})
        old_profiles = previous.get('source_data_configs', {})
        new_profiles = agent.contract.get('source_data_configs', {})
        old_data = {key: value['data_config'] for key, value in old_profiles.items()} or {'default': previous.get('data_config')}
        new_data = {key: value['data_config'] for key, value in new_profiles.items()} or {'default': agent.contract['data_config']}
        if old_data != new_data:
            raise ValueError('Key checkpoint calibration/data profiles differ from V4; use matching source settings')
        load_weights(agent.base, checkpoint, initialize=False)
    else:
        if saved != agent.settings:
            if not initialize:
                raise ValueError('Chunk checkpoint architecture/configuration mismatch')
            changed = {name for name in saved if saved[name] != agent.settings[name]}
            if changed <= {'stage', 'key_loss_weight', 'position_weight', 'rotation_weight', 'gripper_weight',
                           'displacement_weight', 'displacement_scale', 'goal_conditioning'}:
                state = checkpoint['model_state']
                if 'goal_conditioning' in changed:
                    # Same dimensions, different meaning: relearn only the goal
                    # projection while retaining the encoder and trajectory head.
                    state = dict(state)
                    for name, value in agent._net_mod.state_dict().items():
                        if name.startswith('decoder.goal.'):
                            state[name] = value
                agent._net_mod.load_state_dict(state, strict=True)
            else:
                encoder = {key.removeprefix('encoder.'): value for key, value in checkpoint['model_state'].items()
                           if key.startswith('encoder.')}
                agent._net_mod.encoder.load_state_dict(encoder, strict=True)
        else:
            agent._net_mod.load_state_dict(checkpoint['model_state'], strict=True)
    return checkpoint
