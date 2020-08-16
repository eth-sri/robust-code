import argparse
import copy
from abc import ABC
from abc import abstractmethod

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as INIT

from robustcode.models.modules.embeddings import DotProductEmbedding
from robustcode.models.modules.embeddings import MultiEmbedding
from robustcode.models.modules.neural_model_base import NeuralModelBase
from robustcode.models.modules.transformer import EncoderLayer
from robustcode.models.modules.util import clones
from robustcode.models.modules.util import LayerNorm
from robustcode.models.modules.util import PositionwiseFeedForward
from robustcode.util.misc import boolean_string
from robustcode.util.misc import Logger


class MultiHeadAttention(nn.Module):
    "Multi-Head Attention"

    def __init__(self, h, d_model, dropout=0.1):
        """
        h: number of heads
        dim_model: hidden dimension
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        # W_q, W_k, W_v, W_o
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4)
        self.dropout = nn.Dropout(dropout)

    def get(self, x, fields="qkv"):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        ret = {}
        if "q" in fields:
            ret["q"] = self.linears[0](x).view(batch_size, self.h, self.d_k)
        if "k" in fields:
            ret["k"] = self.linears[1](x).view(batch_size, self.h, self.d_k)
        if "v" in fields:
            ret["v"] = self.linears[2](x).view(batch_size, self.h, self.d_k)
        return ret

    def get_edge(self, x, e, field):
        batch_size = x.shape[0]
        if field == "q":
            return self.linears[0](x + e).view(batch_size, self.h, self.d_k)
        if field == "k":
            return self.linears[1](x + e).view(batch_size, self.h, self.d_k)
        assert False

    def get_o(self, x):
        "get output of the multi-head attention"
        batch_size = x.shape[0]
        return self.linears[3](x.view(batch_size, -1))


def src_dot_dst(src_field, dst_field, out_field, attn=None):
    """
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function.
    """

    def func(edges):
        if "emb" not in edges.data:
            return {
                out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                    -1, keepdim=True
                )
            }
        if attn is None:
            return {
                out_field: (
                    (edges.src[src_field] + edges.data["emb"])
                    * (edges.dst[dst_field] + edges.data["emb"])
                ).sum(-1, keepdim=True)
            }
        return {
            out_field: (
                attn.get_edge(edges.src[src_field], edges.data["emb"], src_field)
                * attn.get_edge(edges.dst[dst_field], edges.data["emb"], dst_field)
            ).sum(-1, keepdim=True)
        }

    return func


def scaled_exp(field, c):
    """
    This function applies $exp(x / c)$ for input $x$, which is required by *Scaled Dot-Product Attention* mentioned in the paper.
    """

    def func(edges):
        return {field: torch.exp((edges.data[field] / c).clamp(-10, 10))}

    return func


class HaltingUnit(nn.Module):
    halting_bias_init = -1.0

    def __init__(self, dim_model):
        super(HaltingUnit, self).__init__()
        self.linear = nn.Linear(dim_model, 1)
        self.norm = LayerNorm(dim_model)
        self.reset_parameters()

    def forward(self, x):
        return torch.sigmoid(self.linear(self.norm(x)))

    def reset_parameters(self):
        INIT.constant_(self.linear.bias, self.halting_bias_init)


class UEncoder(nn.Module):
    def __init__(self, layer):
        super(UEncoder, self).__init__()
        self.layer = layer
        self.norm = LayerNorm(layer.size)

    def pre_func(self, fields="qkv"):
        layer = self.layer

        def func(nodes):
            x = nodes.data["x"]
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)

        return func

    def post_func(self):
        layer = self.layer

        def func(nodes):
            x, wv, z = nodes.data["x"], nodes.data["wv"], nodes.data["z"]
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            x = self.norm(x)
            return {"x": x}

        return func


class GraphModel(NeuralModelBase, ABC):
    @staticmethod
    def args():
        parser = argparse.ArgumentParser("GraphModel", add_help=False)
        parser.add_argument(
            "--softmax_type",
            default=NeuralModelBase.SoftmaxType.linear.name,
            choices=[t.name for t in NeuralModelBase.SoftmaxType],
        )
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--d_model", type=int, default=256)
        parser.add_argument(
            "--dot_product_embedding", type=boolean_string, default=False
        )
        return parser

    def __init__(self, config, d_in_vocabs, d_out_vocab):
        super(GraphModel, self).__init__(
            config.d_model, d_out_vocab, softmax_type=config.softmax_type
        )

        self.dropout = nn.Dropout(p=config.dropout)

        if config.dot_product_embedding:
            self.src_embed = nn.Sequential(
                DotProductEmbedding(
                    d_in_vocabs, config.d_model, scaled=False, type="sum"
                )
            )
        else:
            self.src_embed = nn.Sequential(
                MultiEmbedding(d_in_vocabs, config.d_model, scaled=False, type="sum")
            )

        self.graph_features = lambda g: [g.ndata["types"], g.ndata["values"]]
        if len(d_in_vocabs) == 1:
            self.graph_features = lambda g: [g.ndata["types"]]

    def embed(self, g: dgl.DGLGraph):
        return self.dropout(self.src_embed(self.graph_features(g)))

    def forward_with_input_gradient(self, g: dgl.DGLGraph):
        x_onehot = []
        x_embedding = self.src_embed[0].forward(
            self.graph_features(g), x_onehot, require_grad=True
        )
        out = self.forward_with_embed(x_embedding, g)
        return out, x_onehot

    def forward(self, inputs):
        return self.forward_with_embed(self.embed(inputs), inputs)

    @abstractmethod
    def forward_with_embed(self, embed, g: dgl.DGLGraph):
        pass


class UGraphTransformer(GraphModel):
    "Universal Transformer(https://arxiv.org/pdf/1807.03819.pdf) with ACT(https://arxiv.org/pdf/1603.08983.pdf)."
    "Adapted form https://github.com/dmlc/dgl/blob/master/examples/pytorch/transformer/modules/act.py"

    @staticmethod
    def args():
        parser = argparse.ArgumentParser("UGraphTransformerBaseline", add_help=False)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--max_depth", type=int, default=8)
        parser.add_argument("--threshold", type=float, default=0.99)
        parser.add_argument("--act_loss_weight", type=float, default=0.001)
        return parser

    def state_dict(self, **kwargs):
        res = super().state_dict(**kwargs)
        args = {"max_depth": self.max_depth, "threshold": self.threshold}
        res["args"] = args
        return res

    def load_state_dict(self, state_dict, **kwargs):
        for key, value in state_dict["args"].items():
            setattr(self, key, value)
        del state_dict["args"]

        # backward compatibility with previously saved models
        if "out_embed.weight" in state_dict:
            del state_dict["out_embed.weight"]
        super().load_state_dict(state_dict, **kwargs)

    def __init__(self, config, d_in_vocabs, d_out_vocab, **kwargs):
        super(UGraphTransformer, self).__init__(config, d_in_vocabs, d_out_vocab)

        self.max_depth = config.max_depth
        self.threshold = config.threshold
        self.act_loss_weight = config.act_loss_weight
        self.act_loss = None

        c = copy.deepcopy
        multi_attn = MultiHeadAttention(
            config.num_heads, config.d_model, config.dropout
        )
        ff = PositionwiseFeedForward(config.d_model, 2 * config.d_model, config.dropout)

        self.encoder = UEncoder(
            EncoderLayer(config.d_model, multi_attn, c(ff), config.dropout)
        )

        self.h, self.d_k = config.num_heads, config.d_model // config.num_heads
        self.halt_enc = HaltingUnit(self.h * self.d_k)

        self.reset_parameters()
        self.reset_stat()

        # number of iterations for each node
        self.steps = None

    def reset_parameters(self):
        super().reset_parameters()
        self.halt_enc.reset_parameters()

    def reset_stat(self):
        self.stat = [0] * (self.max_depth + 1)

    def print_stat(self):
        values = " ".join(
            "{:6.2f}%".format(
                (100.0 * value / self.stat[0]) if self.stat[0] != 0 else 0
            )
            for value in self.stat
        )
        Logger.debug("nodes entering step: {}".format(values))

    def step_forward(self, nodes):
        x = nodes.data["x"]
        step = nodes.data["step"]
        return {"x": self.dropout(x), "step": step + 1}

    def halt_and_accum(self, end=False):
        halt = self.halt_enc
        threshold = self.threshold

        def func(nodes):
            p = halt(nodes.data["x"])
            sum_p = nodes.data["sum_p"] + p
            active = (sum_p < threshold) & (1 - end)
            _continue = active.float()
            r = nodes.data["r"] * (1 - _continue) + (1 - sum_p) * _continue
            s = (
                nodes.data["s"]
                + ((1 - _continue) * r + _continue * p) * nodes.data["x"]
            )
            return {"p": p, "sum_p": sum_p, "r": r, "s": s, "active": active}

        return func

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(src_dot_dst("k", "q", "score"), eids)
        g.apply_edges(scaled_exp("score", np.sqrt(self.d_k)), eids)
        # Send weighted values to target nodes
        g.send_and_recv(
            eids,
            [fn.src_mul_edge("v", "score", "v"), fn.copy_edge("score", "score")],
            [fn.sum("v", "wv"), fn.sum("score", "z")],
        )

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."
        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # Further calculation after attention mechanism
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)

    def forward_with_embed(self, embed, g: dgl.DGLGraph):
        N = g.number_of_nodes()
        device = next(self.parameters()).device

        if self.max_depth == 0:
            # when the depth is zero we simply return the embeddings
            # maybe we should also apply a non-linearity
            self.steps = torch.zeros(N, dtype=torch.float, device=device)
            return embed

        g.ndata["x"] = embed

        # init step
        g.ndata["s"] = torch.zeros(
            N, self.h * self.d_k, dtype=torch.float, device=device
        )  # accumulated state
        g.ndata["p"] = torch.zeros(
            N, 1, dtype=torch.float, device=device
        )  # halting prob
        g.ndata["r"] = torch.ones(N, 1, dtype=torch.float, device=device)  # remainder
        g.ndata["sum_p"] = torch.zeros(
            N, 1, dtype=torch.float, device=device
        )  # sum of pondering values
        g.ndata["step"] = torch.zeros(N, 1, dtype=torch.long, device=device)  # step
        g.ndata["active"] = torch.ones(N, 1, dtype=torch.bool, device=device)  # active

        # only nodes with at least one incoming edge are active
        g.ndata["active"][:, 0] = g.in_degrees() > 0

        for step in range(self.max_depth):
            pre_func = self.encoder.pre_func("qkv")
            post_func = self.encoder.post_func()
            nodes = g.filter_nodes(lambda v: v.data["active"].view(-1))

            if len(nodes) == 0 or (
                "mask_valid" in g.ndata
                and torch.sum(g.nodes[nodes].data["mask_valid"]) == 0
            ):
                break

            edges = g.filter_edges(lambda e: e.dst["active"].view(-1))

            end = step == self.max_depth - 1
            self.update_graph(
                g,
                edges,
                [(self.step_forward, nodes), (pre_func, nodes)],
                [(post_func, nodes), (self.halt_and_accum(end), nodes)],
            )

        g.ndata["x"] = self.encoder.norm(g.ndata["s"])
        self.act_loss = g.ndata["r"].squeeze(dim=1) * self.act_loss_weight  # ACT loss

        self.steps = g.ndata["step"].squeeze(dim=1)
        steps = torch.masked_select(
            g.ndata["step"].squeeze(dim=1), g.ndata["mask_valid"]
        )
        for step, _ in enumerate(self.stat):
            self.stat[step] += torch.sum(steps >= step).item()

        res = g.ndata.pop("x")
        self.__cleanup(g)
        return res

    """
    Release allocated memory
    """

    def __cleanup(self, g):

        for key in [
            "x",
            "v",
            "wv",
            "z",
            "k",
            "q",
            "s",
            "p",
            "r",
            "sum_p",
            "step",
            "active",
            "score",
        ]:
            if key in g.ndata:
                del g.ndata[key]

        for key in list(g.edata.keys()):
            if key != "type":
                del g.edata[key]
