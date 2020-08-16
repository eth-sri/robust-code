import argparse
import collections
import itertools
from typing import List, Tuple

import torch.nn.functional as F
import dgl
import torch
import torchtext
from torch import nn, Tensor

from robustcode.models.modules.dgl.iterators import EdgeGenerator
from robustcode.models.modules.dgl.utransformer import GraphModel
from robustcode.models.modules.iterators import MiniBatch, BPTTIterator
from robustcode.models.modules.util import LayerNorm
from robustcode.util.misc import boolean_string


class DeepTyperEdgesFieldsGenerator(EdgeGenerator):
    def __init__(self, device):
        """
        Adds edges according to the DeepTyper paper which averages
        over all identifiers with the same name.

        Assumes that the torchtext samples has two fields:
            - 'types' that contains AST types, and
            - 'values' that contains AST values
        """
        super(DeepTyperEdgesFieldsGenerator, self).__init__()
        self.device = device

        # all edges have default type
        self.edge_id("default")

    def add_edges(self, sample: torchtext.data.example.Example, num_nodes: int):
        per_value_usages = collections.defaultdict(list)
        self_loops = []
        for i, ntype, nvalue in zip(itertools.count(), sample.types, sample.values):
            # TODO: hardcoded for JavaScript and TypeScript identifiers
            if ntype != "Identifier":
                self_loops.append(i)
                continue
            per_value_usages[nvalue].append(i)

        edges = []
        # connect all usages together to sum over them
        for usages in per_value_usages.values():
            edges.extend(itertools.product(usages, repeat=2))

        # propagate individual nodes
        edges += [(i, i) for i in self_loops]

        return (
            [s for s, _ in edges],
            [t for _, t in edges],
            {"type": torch.zeros(len(edges), device=self.device)},
        )


class RNNFull(nn.Module):
    def __init__(self, d_model, bidirectional, num_layers, dropout, bptt_len, device):
        super(RNNFull, self).__init__()

        self.batch_first = True
        self.lstm = nn.LSTM(
            d_model,
            d_model if not bidirectional else d_model // 2,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=self.batch_first,
        )
        self.layer_norm = LayerNorm(d_model)

        self.d_model = d_model
        self.bptt_len = bptt_len
        self.device = device

        self.hidden = None

    def forward(self, x):
        outs = []
        self.init_hidden(x.size(dim=0 if self.batch_first else 1))
        for chunk in BPTTIterator(x, self.bptt_len, self.batch_first):
            out, self.hidden = self.lstm(chunk, self.hidden)
            outs.append(out)
            self.repackage_hidden()
        return self.layer_norm(torch.cat(outs, dim=1))

    def init_hidden(self, batch_size):
        b_dim = 2 if self.lstm.bidirectional else 1
        self.hidden = (
            torch.zeros(
                self.lstm.num_layers * b_dim,
                batch_size,
                self.d_model // b_dim,
                device=self.device,
            ),
            torch.zeros(
                self.lstm.num_layers * b_dim,
                batch_size,
                self.d_model // b_dim,
                device=self.device,
            ),
        )

    def repackage_hidden(self):
        self.hidden = (
            torch.autograd.Variable(self.hidden[0]),
            torch.autograd.Variable(self.hidden[1]),
        )


class DeepTyper(GraphModel):
    """
    Implementation of DeepTyper `Deep Learning Type Inference`__

    __ http://vhellendoorn.github.io/PDF/fse2018-j2t.pdf
    """

    @staticmethod
    def args():
        parser = argparse.ArgumentParser("DeepTyper", add_help=False)
        parser.add_argument("--bidirectional", type=boolean_string, default=False)
        parser.add_argument("--num_layers", type=int, default=2)
        return parser

    def __init__(self, config, d_in_vocabs, d_out_vocab, device=None, **kwargs):
        super(DeepTyper, self).__init__(config, d_in_vocabs, d_out_vocab)
        self.device = device

        self.lstm_inner = RNNFull(
            config.d_model,
            config.bidirectional,
            config.num_layers,
            config.dropout,
            config.bptt_len,
            device,
        )

        self.lstm_outer = RNNFull(
            config.d_model,
            config.bidirectional,
            config.num_layers,
            config.dropout,
            config.bptt_len,
            device,
        )

        self.layer_norm = LayerNorm(config.d_model)

    def forward_with_embed(self, embed: Tensor, g: dgl.DGLGraph):
        h = self.lstm_inner(self.dgl_to_rnn_batch(embed, g))
        h = self.rnn_to_dgl_batch(h, g)

        def reduce_func(nodes):
            return {"h": torch.mean(nodes.mailbox["m"], dim=1)}

        g.ndata["h"] = h
        g.update_all(
            message_func=dgl.function.copy_src(src="h", out="m"),
            reduce_func=reduce_func,
        )

        h = F.relu(g.ndata["h"])
        del g.ndata["h"]

        h = self.lstm_outer(self.dgl_to_rnn_batch(h, g))
        h = self.rnn_to_dgl_batch(h, g)
        return h

    def dgl_to_rnn_batch(self, h: Tensor, g: dgl.DGLGraph):
        """
        Converts batch used as GNN input to a batch used for RNNs

        Args:
            h: Tensor of size N x D, where N is number of elements in the graph and D is their dimensionality
            g: batched graph

        Returns:
            out: Tensor of size B x M x D, where B is the number of batched graphs and M is the largest graph size

        """
        h = h.split(g.batch_num_nodes)
        h = torch.nn.utils.rnn.pad_sequence(h, batch_first=True)
        return h

    def rnn_to_dgl_batch(self, h: Tensor, g: dgl.DGLGraph):
        """
        Reverse operation of `dgl_to_rnn_batch`
        """
        return torch.cat([x[:num_nodes] for x, num_nodes in zip(h, g.batch_num_nodes)])
