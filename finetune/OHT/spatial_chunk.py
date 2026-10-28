"""TCP-local 3D relative attention for regression and conditional flow matching.

Only observed geometry and the predicted key pose condition the policy. Flow
training uses noisy future poses; deployment samples a fresh trajectory, never
receiving clean futures, future validity or a required key-goal endpoint.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .chunk_model import ChunkDecoder, quaternion_matrix, rotation_6d
from .chunk_config import chunk_scale


def time_features(value, width=32):
    frequencies = torch.exp(torch.arange(width // 2, device=value.device, dtype=value.dtype)
                            * (-math.log(10000.) / (width // 2 - 1)))
    phase = value[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), -1)


def to_local(points, state):
    rotation = quaternion_matrix(state[:, 3:7])
    return torch.einsum('bij,bnj->bni', rotation.transpose(-1, -2), points - state[:, None, :3])


class RelativeAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.query, self.key, self.value, self.output = (nn.Linear(dim, dim) for _ in range(4))
        self.bias = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, heads))

    def forward(self, query, memory, query_xyz, memory_xyz, memory_valid):
        batch, count, dim = query.shape
        split = lambda value: value.reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        delta = memory_xyz[:, None] - query_xyz[:, :, None]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        # Bounded direction plus logarithmic distance retain a broad scene
        # context without raw large cube coordinates dominating attention.
        relative = torch.cat((delta / (1 + distance), distance.log1p()), -1)
        bias = self.bias(relative).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~memory_valid[:, None, None], float('-inf'))
        value = F.scaled_dot_product_attention(split(self.query(query)), split(self.key(memory)),
                    split(self.value(memory)), attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.)
        return self.output(value.transpose(1, 2).reshape(batch, count, dim))


class SpatialBlock(nn.Module):
    def __init__(self, dim, heads, dropout, adaln=False):
        super().__init__()
        self.self_attention = RelativeAttention(dim, heads, dropout)
        self.cross_attention = RelativeAttention(dim, heads, dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(dim, elementwise_affine=not adaln) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim)) if adaln else None
        if self.modulation is not None:
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, query, xyz, memory, positions, valid, time=None):
        if self.modulation is not None:
            if time is None:
                raise ValueError('AdaLN block requires a flow-time condition')
            shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = (
                value[:, None] for value in self.modulation(time).chunk(9, dim=-1))
            normalized = self.norms[0](query) * (1 + scale1) + shift1
            query = query + gate1 * self.self_attention(normalized, normalized, xyz, xyz,
                                            torch.ones_like(xyz[..., 0], dtype=torch.bool))
            normalized = self.norms[1](query) * (1 + scale2) + shift2
            query = query + gate2 * self.cross_attention(normalized, memory, xyz, positions, valid)
            normalized = self.norms[2](query) * (1 + scale3) + shift3
            return query + gate3 * self.ffn(normalized)
        normalized = self.norms[0](query)
        query = query + self.self_attention(normalized, normalized, xyz, xyz,
                                           torch.ones_like(xyz[..., 0], dtype=torch.bool))
        query = query + self.cross_attention(self.norms[1](query), memory, xyz, positions, valid)
        return query + self.ffn(self.norms[2](query))


class SpatialChunkDecoder(nn.Module):
    goal_condition = ChunkDecoder.goal_condition

    def __init__(self, feature_dim, settings, num_views, action_dt):
        super().__init__()
        self.settings, self.num_views, self.action_dt = settings, num_views, action_dt
        chunk_scale(settings, 'action_position_scale')
        self.is_flow = settings['decoder_type'] == 'flow_matching'
        dim = settings['hidden_dim']
        self.visual, self.state, self.goal = nn.Linear(feature_dim, dim), nn.Linear(14, dim), nn.Linear(8, dim)
        self.scale_embedding = nn.Parameter(torch.zeros(2, dim))
        self.view_embedding = nn.Embedding(num_views, dim)
        self.queries = nn.Parameter(torch.randn(settings['horizon'], dim) * .02)
        self.action = nn.Linear(9, dim)
        self.flow_time = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.tick_time = nn.Sequential(nn.Linear(32, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(SpatialBlock(dim, settings['heads'], settings['dropout'], settings.get('adaln', False))
                                    for _ in range(settings['layers']))
        kernel = settings.get('temporal_kernel_size', 1)
        self.temporal = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim) if kernel > 1 else None
        if self.temporal is not None:
            nn.init.zeros_(self.temporal.weight)
            nn.init.zeros_(self.temporal.bias)
        self.head = nn.Linear(dim, 10)
        nn.init.normal_(self.head.weight, std=.001)
        nn.init.zeros_(self.head.bias)
        if not self.is_flow:
            with torch.no_grad():
                self.head.bias[3:9] = torch.tensor([1., 0., 0., 0., 1., 0.])

    def memory(self, coarse, fine, state, goal, coarse_xyz, fine_xyz, coarse_valid, fine_valid):
        features, positions, validity = [], [], []
        scale, dtype = chunk_scale(self.settings, 'geometry_scale'), self.visual.weight.dtype
        for index, (tokens, xyz, valid) in enumerate(((coarse, coarse_xyz, coarse_valid),
                                                     (fine, fine_xyz, fine_valid))):
            if xyz.shape != (*tokens.shape[:2], 3) or valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
                raise ValueError('Spatial chunk tokens require aligned XYZ and boolean validity')
            if tokens.shape[1] % self.num_views:
                raise ValueError('Spatial chunk tokens must retain complete view/patch ordering')
            if not torch.isfinite(xyz[valid]).all() or not torch.isfinite(tokens[valid]).all():
                raise ValueError('Valid spatial chunk tokens must be finite')
            xyz = torch.where(valid[..., None], xyz, state[:, None, :3])
            tokens = torch.where(valid[..., None], tokens, 0.)
            views = torch.arange(self.num_views, device=tokens.device).repeat_interleave(tokens.shape[1] // self.num_views)
            features.append(self.visual(tokens.to(dtype)) + self.scale_embedding[index] + self.view_embedding(views)[None])
            positions.append(to_local(xyz, state) / scale)
            validity.append(valid)
        local_state = state.clone()
        local_state[:, :7] = state.new_tensor([0., 0., 0., 0., 0., 0., 1.])
        relative_goal = self.goal_condition(state, goal)
        features.extend((self.state(local_state.to(dtype))[:, None], self.goal(relative_goal.to(dtype))[:, None]))
        positions.extend((torch.zeros_like(state[:, None, :3]), relative_goal[:, None, :3]))
        validity.extend((torch.ones_like(state[:, :1], dtype=torch.bool),) * 2)
        return torch.cat(features, 1), torch.cat(positions, 1).to(dtype), torch.cat(validity, 1)

    def field(self, pose, tau, memory):
        features, positions, valid = memory
        dtype = features.dtype
        pose = pose.to(dtype)
        ticks = torch.arange(1, pose.shape[1] + 1, device=pose.device, dtype=dtype) * self.action_dt
        time = self.flow_time(time_features(tau.to(dtype)))
        query = (self.action(pose) + self.queries[None] + self.tick_time(time_features(ticks))[None]
                 + time[:, None])
        # Flow tokens are action-normalized. Relative attention must compare
        # them to scene/key coordinates in the *geometry* normalization instead.
        xyz = pose[..., :3] * (chunk_scale(self.settings, 'action_position_scale') /
                                chunk_scale(self.settings, 'geometry_scale'))
        for block in self.blocks:
            query = block(query, xyz, features, positions, valid, time)
        if self.temporal is not None:
            query = query + self.temporal(query.transpose(1, 2)).transpose(1, 2)
        raw = self.head(query)
        return raw[..., :9], raw[..., 9]

    def decode(self, pose, grip, state):
        rotation = quaternion_matrix(state[:, 3:7])
        position = torch.einsum('bij,bkj->bki', rotation, pose[..., :3] * chunk_scale(self.settings, 'action_position_scale'))
        return (position + state[:, None, :3], rotation[:, None] @ rotation_6d(pose[..., 3:9]), grip)

    def encode_target(self, target, state):
        local_rotation = quaternion_matrix(state[:, 3:7]).transpose(-1, -2)[:, None] @ quaternion_matrix(target[..., 3:7])
        return torch.cat((to_local(target[..., :3], state) / chunk_scale(self.settings, 'action_position_scale'),
                          local_rotation[..., :, 0], local_rotation[..., :, 1]), -1)

    def sample(self, memory, state):
        batch, horizon = state.shape[0], self.settings['horizon']
        if not self.is_flow:
            pose = state.new_zeros(batch, horizon, 9)
            pose[..., 3], pose[..., 7] = 1., 1.
            raw, grip = self.field(pose, state.new_ones(batch), memory)
            return self.decode(raw, grip, state)
        generator = None
        if self.settings['flow_inference_seed'] is not None:
            generator = torch.Generator(device=state.device).manual_seed(self.settings['flow_inference_seed'])
        count = 1 if generator is not None else batch
        pose = torch.randn(count, horizon, 9, device=state.device, dtype=state.dtype, generator=generator).expand(batch, -1, -1)
        steps = self.settings['flow_steps']
        for index in range(steps):
            velocity, _ = self.field(pose, state.new_full((batch,), index / steps), memory)
            pose = pose + velocity / steps
        _, grip = self.field(pose, state.new_ones(batch), memory)
        return self.decode(pose, grip, state)

    def forward(self, coarse, fine, state, goal, coarse_xyz, fine_xyz, coarse_valid, fine_valid):
        memory = self.memory(coarse, fine, state, goal, coarse_xyz, fine_xyz, coarse_valid, fine_valid)
        return self.sample(memory, state)

    def sample_flow_time(self, state):
        uniform = torch.rand(len(state), device=state.device, dtype=state.dtype)
        if self.settings.get('flow_time_sampling', 'uniform') == 'beta_high_noise':
            # pi0-style shifted Beta(1.5, 1): tau=0 is the high-noise end.
            return .999 * (1 - uniform.pow(1 / 1.5))
        return uniform

    def training_prediction(self, coarse, fine, state, goal, coarse_xyz, fine_xyz, coarse_valid, fine_valid, *, target, valid):
        memory = self.memory(coarse, fine, state, goal, coarse_xyz, fine_xyz, coarse_valid, fine_valid)
        if not self.is_flow:
            return self.sample(memory, state), {}
        target, valid = target[:, :self.settings['horizon']], valid[:, :self.settings['horizon']]
        if target.shape != (len(state), self.settings['horizon'], 8) or valid.shape != target.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('Flow targets require [B,K,8] and boolean [B,K]')
        if not valid.any() or not torch.isfinite(target[valid]).all():
            raise ValueError('Flow requires valid finite future supervision')
        safe_target = torch.where(valid[..., None], target, torch.cat((state[:, None, :7], state.new_zeros(len(state), 1, 1)), -1))
        clean = self.encode_target(safe_target, state)
        noise, tau = torch.randn_like(clean), self.sample_flow_time(state)
        mixed = (1 - tau[:, None, None]) * noise + tau[:, None, None] * clean
        mixed = torch.where(valid[..., None], mixed, noise)
        velocity, grip = self.field(mixed, tau, memory)
        expected = clean - noise
        position_loss = F.mse_loss(velocity[..., :3][valid], expected[..., :3][valid])
        rotation_loss = F.mse_loss(velocity[..., 3:9][valid], expected[..., 3:9][valid])
        flow_loss = self.settings['position_weight'] * position_loss + self.settings['rotation_weight'] * rotation_loss
        reconstruction = mixed + (1 - tau[:, None, None]) * velocity
        bins = {}
        for name, lower, upper in (('high_noise', 0., 1 / 3), ('mid_noise', 1 / 3, 2 / 3), ('low_noise', 2 / 3, 1.)):
            selected = valid & (tau[:, None] >= lower) & (tau[:, None] < upper)
            bins[f'chunk_flow_tau_{name}_count'] = int(selected.sum().item())
            bins[f'chunk_flow_tau_{name}_position_loss'] = (
                F.mse_loss(velocity[..., :3][selected], expected[..., :3][selected]).detach().item()
                if selected.any() else 0.)
        # Diagnostics and TensorBoard use an independent deployment sample.
        # Clean/noisy futures above must not make policy curves look accurate.
        training = self.training
        try:
            self.eval()
            with torch.no_grad():
                sampled = self.sample(tuple(value.detach() for value in memory), state)
        finally:
            self.train(training)
        return self.decode(reconstruction, grip, state), dict(
            flow_loss=flow_loss, sampled_prediction=sampled,
            chunk_flow_position_loss=position_loss.detach().item(),
            chunk_flow_rotation_loss=rotation_loss.detach().item(), **bins)
