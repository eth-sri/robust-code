import argparse
import copy
import math
import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Random:
    @staticmethod
    def seed(seed):
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def args():
        parser = argparse.ArgumentParser("Random", add_help=False)
        parser.add_argument("--seed", type=int, default=42, help="Random Seed")
        return parser


class ModelEval:
    """
    Runs in the evaluation mode and restores the original mode on exit

    Usage:
    with ModelEval(model):
       ...
    """

    def __init__(self, model):
        self.model = model
        self.training = None

    def __enter__(self):
        self.training = self.model.training
        if self.training:
            self.model.eval()
        return None

    def __exit__(self, type, value, traceback):
        if self.training:
            self.model.train()


class AdaptiveLogSoftmax(nn.AdaptiveLogSoftmaxWithLoss):
    """
    Wrapper for AdaptiveLogSoftmax class that allows taking batched inputs
    """

    def predict(self, input):
        if len(input.size()) == 2:
            return super(AdaptiveLogSoftmax, self).predict(input)
        nbatches = input.size(dim=1)
        return (
            super(AdaptiveLogSoftmax, self)
            .predict(input.view(-1, input.size(dim=-1)))
            .view(-1, nbatches)
        )

    def log_prob(self, input):
        if len(input.size()) == 2:
            return super(AdaptiveLogSoftmax, self).log_prob(input)
        nbatches = input.size(dim=1)
        return (
            super(AdaptiveLogSoftmax, self)
            .log_prob(input.view(-1, input.size(dim=-1)))
            .view(-1, nbatches)
        )

    def topk(self, input, k):
        assert False, "Not implemented!"


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, d_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        return self.proj(x)
        # return F.log_softmax(self.proj(x), dim=-1)

    def predict(self, x):
        return self(x).argmax(dim=-1)

    def topk(self, x, k):
        return self.log_prob(x).topk(k=k, dim=-1)

    def log_prob(self, x):
        return F.log_softmax(self(x), dim=-1)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, use_offset=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

        self.offset = 0
        self.use_offset = use_offset

    def at(self, offset):
        if isinstance(offset, torch.Tensor):
            return torch.index_select(self.pe, 0, offset)
        return self.pe[offset, :]

    def forward(self, x):
        offset = 0 if not self.use_offset else self.offset
        x = x + torch.autograd.Variable(
            self.pe[offset : offset + x.size(0), :], requires_grad=False
        )
        offset += x.size(0)
        return self.dropout(x)

    def init_hidden(self, batch_size):
        self.offset = 0


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
