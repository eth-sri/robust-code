import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from robustcode.models.modules.sparse_activations import Sparsemax
from robustcode.models.modules.util import clones


def attention(query, key, value, dropout, mask=None, softmax=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    if softmax is not None:
        p_attn = softmax(scores, dim=-1)
    else:
        p_attn = F.softmax(scores, dim=-1)
    p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, type="softmax"):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

        if type == "softmax":
            self.softmax = F.softmax
        if type == "sparsemax":
            self.softmax = Sparsemax()

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(-1, nbatches, self.h, self.d_k).transpose(0, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = attention(
            query, key, value, self.dropout, mask=mask, softmax=self.softmax
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(0, 2).contiguous().view(-1, nbatches, self.h * self.d_k)
        return self.linears[-1](x)
