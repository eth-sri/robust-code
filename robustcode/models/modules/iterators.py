import collections
import copy
from typing import Iterable

import dgl
import torch
import torchtext


class MiniBatch(
    collections.namedtuple(
        "MiniBatch", ["X", "Y", "lengths", "masks", "P", "data", "ids"]
    )
):
    __slots__ = ()

    def clone(self):
        assert self.P is None, "clone not implemented for field P."
        if isinstance(self.X, list):
            X = [x.detach().clone() for x in self.X]
            M = {key: value.detach().clone() for key, value in self.masks.items()}
        elif isinstance(self.X, torch.Tensor):
            X = self.X.detach().clone()
        elif isinstance(self.X, dgl.DGLGraph):
            X = dgl.batch(dgl.unbatch(self.X))
            M = {mask: X.ndata[mask] for mask in self.masks.keys()}
        else:
            assert False, "unhandled type to clone: {}".format(type(self.X))

        return MiniBatch(
            X,
            self.Y.detach().clone(),  # tensor
            copy.copy(self.lengths),  # list
            M,  # dict of tensors
            None,  # ??
            copy.deepcopy(self.data) if self.data is not None else None,  # dist
            copy.copy(self.ids) if self.ids is not None else None,  # list
        )


class SequenceIterator:
    def __init__(
        self,
        it: torchtext.data.BucketIterator,
        input_fields: Iterable[str],
        target_field: str,
        mask_fields: Iterable[str] = None,
    ):
        """

        Args:
            it: iterator to wrap around
            input_fields: list of input fields names wrap as X
            target_field: target field name to wrap as Y
            mask_fields: list of mask fields to wrap as masks
        """
        self.it = it
        self.input_fields = input_fields
        self.target_field = target_field
        self.mask_fields = mask_fields if mask_fields is not None else []

    def init_epoch(self):
        self.it.init_epoch()

    def __len__(self):
        return len(self.it)

    def __iter__(self):
        for batch in self.it:
            """
             assumes atleast one the field was created with include_lengths=True, e.g.:
             torchtext.data.Field(sequential=True, include_lengths=True, eos_token='</s>', init_token='<s>')
            """
            L = None
            Y = getattr(batch, self.target_field)
            if isinstance(Y, tuple):
                Y = Y[0]

            X = []
            for field in self.input_fields:
                data = getattr(batch, field)
                """
                in case field was created with lengths=True it will be a tuple (values, lengths)
                since we already have the length from target, we drop the lengths here
                """
                X.append(data[0] if isinstance(data, tuple) else data)
                if L is None and isinstance(data, tuple):
                    L = data[1]

            M = {
                mask_name: getattr(batch, mask_name).bool()
                for mask_name in self.mask_fields
            }

            yield MiniBatch(X, Y, L.tolist(), M, None, None, batch.id.tolist())


class BPTTIterator:
    def __init__(self, x: torch.Tensor, bptt_len: int, batch_first: bool = False):
        """
        Iterates over x in chunks of bptt_len

        Args:
            x: tensor of values of shape B x * if batch_first is True or N x B x * of False
            bptt_len: size of chunks to generate
        """
        self.x = x
        self.bptt_len = bptt_len
        self.batch_first = batch_first

    def __iter__(self):
        length_dim = 1 if self.batch_first else 0
        size = self.x.size(length_dim)
        for i in range(0, size, self.bptt_len):
            chunk_len = min(self.bptt_len, size - i)
            chunk = self.x.narrow(length_dim, i, chunk_len)
            yield chunk
