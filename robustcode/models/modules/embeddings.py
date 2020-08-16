import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from robustcode.models.modules.util import LayerNorm


class MultiEmbedding(nn.Module):
    def __init__(self, d_in_vocabs, d_embedding, scaled=False, type="concat"):
        super(MultiEmbedding, self).__init__()
        if not isinstance(d_in_vocabs, list):
            d_in_vocabs = [d_in_vocabs]

        self.factor = len(d_in_vocabs) if type == "concat" else 1
        # emb_cls = nn.Embedding if not delta_grad else EmbeddingWithGrad
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(d_vocab, d_embedding // self.factor)
                for d_vocab in d_in_vocabs
            ]
        )

        self.norm = LayerNorm(d_embedding)
        self.s_factor = math.sqrt(d_embedding // len(d_in_vocabs)) if scaled else 1
        self.type = type
        self.d_in_vocabs = d_in_vocabs

    def forward(self, x, initial_embeddings=None):
        """
        Embed multiple tokens at once and add their embeddings.

        :param initial_embeddings:  Optional. If a list is provided, the initial embeddings will be saved here. Useful for gradient computations.
        """
        if not isinstance(x, list):
            return self.embeddings[0](x) * self.s_factor
        else:
            assert len(x) == len(self.embeddings)
            if initial_embeddings is None:
                initial_embeddings = []
            initial_embeddings += [
                embed_fn(vec) for embed_fn, vec in zip(self.embeddings, x)
            ]
            assert len(initial_embeddings) == len(x)
            if self.type == "concat":
                return torch.cat(
                    [embedding * self.s_factor for embedding in initial_embeddings],
                    dim=-1,
                )
            elif self.type == "sum":
                res = initial_embeddings[0] * self.s_factor
                for embedding in initial_embeddings[1:]:
                    res.add_(embedding * self.s_factor)
                return self.norm(res)

            assert False


class DotProductEmbedding(MultiEmbedding):
    def _dot_embed(self, input, which, require_grad=False):
        """
        Embeds a tensor using matrix multiplication with embedding weights.
        Used to obtain differentiable embeddings.

        :param input:   tensor of vocab indices
        :param which:   int, which embedding to use
        :param require_grad:    True to enable gradient w.r.t. input
        :returns:   tuple(embedded_input, onehot_input)
        """
        assert 0 <= which < len(self.d_in_vocabs)
        X = F.one_hot(input, num_classes=self.d_in_vocabs[which]).float()
        assert X.grad_fn is None
        if require_grad:
            X.requires_grad_()
        W = self.embeddings[which].weight
        return (torch.matmul(X, W) * self.s_factor, X)

    def forward(self, x, leaf_ops=None, require_grad=False):
        if leaf_ops is None:
            leaf_ops = []
        assert len(leaf_ops) == 0

        if not isinstance(x, list):
            x_embedding, x_onehot = self._dot_embed(x, 0, require_grad)
            leaf_ops.append(x_onehot)
            return x_embedding * self.s_factor

        assert len(x) == len(self.embeddings)
        embeddings = []

        for i, x_i in enumerate(x):
            x_embedding, x_onehot = self._dot_embed(x_i, i, require_grad)
            leaf_ops.append(x_onehot)
            embeddings.append(x_embedding * self.s_factor)

        if self.type == "concat":
            return torch.cat(embeddings, dim=-1)
        elif self.type == "sum":
            res = embeddings[0]
            for embedding in embeddings[1:]:
                res.add_(embedding)
            return self.norm(res)
