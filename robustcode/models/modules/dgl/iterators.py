import logging
import random
from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import List
from typing import Union

import dgl
import torch
import torchtext

from robustcode.models.modules.iterators import MiniBatch


class EdgeGenerator(ABC):
    """
    Base class representing edge generators for graph models
    """

    def __init__(self):
        self.id_to_edge_type = []
        self.edge_type_to_id = {}

    def edge_id(self, edge_type: str) -> int:
        """

        Args:
            edge_type: string representation of the edge type

        Returns: numeric value corresponding to the edge_type

        """
        if edge_type in self.edge_type_to_id:
            return self.edge_type_to_id[edge_type]
        idx = len(self.edge_type_to_id)
        self.edge_type_to_id[edge_type] = idx
        self.id_to_edge_type.append(edge_type)
        return idx

    def edge_str(self, edge_id: int) -> str:
        """
        Reverse operations from edge_id
        """
        assert 0 <= edge_id < len(self.id_to_edge_type)
        return self.id_to_edge_type[edge_id]

    def num_edge_types(self) -> int:
        """
        Returns: number of edge types
        """
        return len(self.id_to_edge_type)

    @abstractmethod
    def add_edges(self, sample: torchtext.data.example.Example, num_nodes: int):
        """
        Computes edges corresponding to the given sample

        Args:
            sample: torchtext sample
            num_nodes: number of nodes in the graph

        Returns: a tuple (u, v, edge_types) of lists with equal length:
                    - u is a list of source nodes,
                    - v is a list of target nodes,
                    - edge_types is list of numericalized types

        For example, returning ([1,2], [2, 4], [5, 6]) denotes that:
            - an edge between (1,2) is added with type 5
            - an edge between (2,4) is added with type 6

        """
        pass


class EdgesFieldsGenerator(EdgeGenerator):
    def __init__(self, fields: Iterable[str], device, self_loop=True, window_size=0):
        """

        Args:
            fields: list of field names that contain edges. For a field name `f`,
            the torchtext sample is expected to have two attributes f_src and f_tgt.
            The name of `f` is used as the edge type.

            device: device where to allocate the edge data
            self_loop: if set to True, self loops are added
            window_size: the size of window edges to add.

        For example, to encode basic ast edges the torchtext sample can have following fields:
            - 'parent_src': [1,2,3]
            - 'parent_tgt': [2,3,4]

        Then, the edge generator is invoked with fields=['parent'] to add edges[(1,2), (2,3), (3,4)]
        """
        super(EdgesFieldsGenerator, self).__init__()
        self.fields = fields
        self.self_loop = self_loop
        self.device = device
        self.window_size = window_size

    def add_edges(self, sample: torchtext.data.example.Example, num_nodes: int):
        edges_src = []
        edges_tgt = []
        edges_ids = []
        if self.self_loop:
            edges_src = [i for i in range(num_nodes)]
            edges_tgt = [i for i in range(num_nodes)]
            edge_id = self.edge_id("self")
            edges_ids = [edge_id for _ in range(num_nodes)]

        unique_edges = set()

        def exists(u, v):
            if u == v:
                return True
            if u > v:
                u, v = v, u
            if (u, v) in unique_edges:
                return True
            unique_edges.add((u, v))
            return False

        for edge_name in self.fields:
            src = getattr(sample, edge_name + "_src")
            tgt = getattr(sample, edge_name + "_tgt")
            edge_id = self.edge_id(edge_name)

            for i, j in zip(src, tgt):
                if exists(i, j):
                    continue

                edges_src += [i, j]
                edges_tgt += [j, i]
                edges_ids += [edge_id, edge_id]

        """
        add all edges in the range [i - window_size, i + window_size]
        TODO: optimize by generating edges in batches
        """
        # assert self.window_size == 0
        if self.window_size != 0:
            for i in range(num_nodes):
                for j in range(
                    max(0, i - self.window_size),
                    min(num_nodes, i + self.window_size + 1),
                ):
                    if exists(i, j):
                        continue
                    edges_src.append(i)
                    edges_tgt.append(j)
                    edges_ids.append(self.edge_id("window_{}".format(j - i)))

        return (
            edges_src,
            edges_tgt,
            {"type": torch.tensor(edges_ids, device=self.device)},
        )


class GraphBatchIterator:
    def __init__(
        self,
        dataset: torchtext.data.Dataset,
        input_fields: List[str],
        target_field: str,
        batch_size: int,
        edge_gen,
        mask_fields: List[str] = None,
        device=None,
        sort_key=None,
        shuffle=True,
        cached=False,
        per_node_target=True,
    ):
        if isinstance(device, int):
            logging.warning(
                "The `device` argument should be set by using `torch.device`"
                + " or passing a string as an argument. This behavior will be"
                + " deprecated soon and currently defaults to cpu."
            )
            device = None
        self.edge_gen = edge_gen
        self.dataset = dataset
        self.device = device
        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key
        self.batch_size = batch_size
        # can be extended by using the third argument to obtain batches of roughly same size (of nodes)
        self.batches: List[List[torchtext.data.example.Example]] = list(
            torchtext.data.batch(
                sorted(dataset, key=self.sort_key), self.batch_size, None
            )
        )

        self.input_fields = input_fields
        self.target_field = target_field
        self.mask_fields = list(mask_fields) if mask_fields is not None else []

        self.shuffle = shuffle
        self.per_node_target = per_node_target

        self.cached = False
        if cached:
            self.batches = [batch for batch in self]
            self.cached = True

    def init_epoch(self):
        if self.shuffle:
            random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.cached:
            for batch in self.batches:
                yield batch
            return

        for batch in self.batches:
            trees = []
            L = []
            ids = []
            for sample in batch:
                ids.append(sample.id)
                data, len = self.__node_data(sample)
                u, v, edata = self.edge_gen.add_edges(sample, len)
                L.append(len)

                g = dgl.DGLGraph()
                g.add_nodes(len, data=data)
                g.add_edges(u, v, data=edata)

                trees.append(g)

            ids = self.dataset.fields["id"].numericalize(ids)
            # batch by merging into one graph
            batch_trees = dgl.batch(trees)

            Y = self.__compute_target(batch)
            M = {name: batch_trees.ndata[name] for name in self.mask_fields}
            yield MiniBatch(batch_trees, Y, L, M, None, None, ids.tolist())

    def __compute_target(self, batch):
        if self.per_node_target:
            Y = torch.cat(
                list(
                    self.__numericalize_field(sample, self.target_field)
                    for sample in batch
                )
            )
            return Y

        for sample in batch:
            print(getattr(sample, self.target_field))
        Y = self.__numericalize_field([sample for sample in batch], self.target_field)
        return Y

    def __numericalize_field(
        self,
        samples: Union[
            torchtext.data.example.Example, List[torchtext.data.example.Example]
        ],
        name: str,
    ):
        if not isinstance(samples, list):
            samples = [samples]
        data = [getattr(sample, name) for sample in samples]
        field: torchtext.data.Field = self.dataset.fields[name]

        # we are using single samples (not batches) so turn off include_lengths
        include_lengths = field.include_lengths
        field.include_lengths = False
        if len(data) >= 1:
            data = field.pad(data)
        data = field.numericalize(data, self.device)
        field.include_lengths = include_lengths

        if len(samples) == 1:
            # torchtext returns a batch with a single example, remove the extra dimension
            return data.squeeze(dim=1)
        return data

    def __node_data(self, sample: torchtext.data.example.Example):
        data = {}
        assert len(self.input_fields) > 0
        length = len(getattr(sample, self.input_fields[0]))
        for field in self.input_fields:
            data[field] = self.__numericalize_field(sample, field)
            assert data[field].size(0) == length
        for field in self.mask_fields:
            data[field] = self.__numericalize_field(sample, field).bool()
            assert data[field].size(0) == length

        return data, length
