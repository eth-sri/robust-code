import argparse

import torch
from torch import nn

from robustcode.models.modules.attention import MultiHeadedAttention
from robustcode.models.modules.embeddings import MultiEmbedding, DotProductEmbedding
from robustcode.models.modules.iterators import BPTTIterator
from robustcode.models.modules.neural_model_base import NeuralModelBase
from robustcode.models.modules.util import LayerNorm
from robustcode.util.misc import boolean_string


class RNN(NeuralModelBase):
    @staticmethod
    def args():
        parser = argparse.ArgumentParser("RNN", add_help=False)
        parser.add_argument(
            "--softmax_type",
            default=NeuralModelBase.SoftmaxType.linear.name,
            choices=[t.name for t in NeuralModelBase.SoftmaxType],
        )
        parser.add_argument("--d_model", type=int, default=64)
        parser.add_argument("--bidirectional", type=boolean_string, default=False)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument(
            "--dot_product_embedding", type=boolean_string, default=False
        )
        parser.add_argument("--bptt_len", type=int)
        return parser

    def __init__(self, config, d_in_vocabs, d_out_vocab, device=None, **kwargs):
        super(RNN, self).__init__(
            config.d_model, d_out_vocab, softmax_type=config.softmax_type
        )
        self.device = device
        self.hidden = None

        if config.dot_product_embedding:
            self.embeddings = DotProductEmbedding(
                d_in_vocabs, config.d_model, type="sum"
            )
        else:
            self.embeddings = MultiEmbedding(d_in_vocabs, config.d_model, type="sum")

        # if this class is used with SequenceIterator then batch_first = False
        self.batch_first = False
        self.lstm = nn.LSTM(
            config.d_model,
            config.d_model if not config.bidirectional else config.d_model // 2,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=self.batch_first,
        )

        self.layer_norm = LayerNorm(config.d_model)
        self.bptt_len = config.bptt_len

    def embed(self, inputs):
        return self.embeddings(inputs)

    def forward_with_embed(self, embed, inputs):
        outs = []
        self.init_hidden(embed.size(dim=0 if self.batch_first else 1))
        for chunk in BPTTIterator(embed, self.bptt_len, self.batch_first):
            out, self.hidden = self.lstm(chunk, self.hidden)
            outs.append(out)
            self.repackage_hidden()

        return self.layer_norm(torch.cat(outs, dim=0))

    def forward(self, inputs):
        return self.forward_with_embed(self.embed(inputs), inputs)

    def forward_with_input_gradient(self, inputs):
        x_onehot = []
        x_embedding = self.embeddings.forward(inputs, x_onehot, require_grad=True)
        out = self.forward_with_embed(x_embedding, inputs)
        return out, x_onehot

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
        # self.init_hidden(self.hidden[0].size(dim=1))
        self.hidden = (
            torch.autograd.Variable(self.hidden[0]),
            torch.autograd.Variable(self.hidden[1]),
        )


class RNNWithAttention(RNN):
    @staticmethod
    def args():
        parser = argparse.ArgumentParser(
            "RNNWithAttention", add_help=False, parents=[RNN.args()]
        )
        parser.add_argument("--num_heads", type=int, default=4)
        return parser

    def __init__(
        self, config, d_in_vocabs, d_out_vocab, device=None, pad_token_id=None
    ):
        super(RNNWithAttention, self).__init__(config, d_in_vocabs, d_out_vocab, device)
        self.pad_token_id = pad_token_id
        assert pad_token_id is not None

        # Attention
        self.attn = MultiHeadedAttention(
            config.num_heads, config.d_model, config.dropout
        )

    def forward_with_embed(self, embed, inputs):
        x = super().forward_with_embed(embed, inputs)

        src_mask = (
            ((inputs[0] if isinstance(inputs, list) else inputs) != self.pad_token_id)
            .transpose(0, 1)
            .to(self.device)
        )

        x = self.attn(x, x, x, src_mask)
        return x

    def forward(self, inputs):
        return self.forward_with_embed(self.embed(inputs), inputs)
