"""Optional coarse semantic hints for an otherwise independent local selector."""

import math

import torch
from torch import nn


class SoftCoarseQueryConditioner(nn.Module):
    """Read soft T/R tokens without importing confidence, geometry or masks.

    Local unordered queries attend to role-labelled coarse tokens. Reference
    values are softly weighted by non-NULL probability; NULL is never a hard
    local decision. A zero output projection exactly preserves local queries
    at initialization, while the local decoder always reads its own features.
    """

    def __init__(self, dim):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.role_embedding = nn.Parameter(torch.randn(2, dim) * .02)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, queries, role_tokens, reference_null_probability):
        batch, _, dim = queries.shape
        if role_tokens.shape != (batch, 2, dim):
            raise ValueError('coarse role tokens must have shape [B,2,D]')
        if reference_null_probability.shape != (batch,):
            raise ValueError('coarse Reference NULL probability must have shape [B]')
        context = self.context_norm(role_tokens.to(queries)) + self.role_embedding
        scores = torch.einsum('bkd,brd->bkr', self.query(self.query_norm(queries)),
                              self.key(context)) / math.sqrt(dim)
        non_null = 1 - reference_null_probability.to(queries).clamp(0, 1)
        weights = torch.stack((torch.ones_like(non_null), non_null), dim=1)
        values = self.value(context) * weights[..., None]
        hint = torch.einsum('bkr,brd->bkd', scores.softmax(-1), values)
        return queries + self.output(hint)
