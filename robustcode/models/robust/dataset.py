import argparse
import itertools
import os
from enum import Enum
from types import SimpleNamespace
from typing import List

import numpy
import torch
import torchtext

from robustcode.analysis.graph import AstTree
from robustcode.analysis.graph import TypeScriptGraphAnalyzer
from robustcode.datasets.dataset import Config
from robustcode.datasets.dataset import TabularCompressedDataset
from robustcode.dedup import fastsim
from robustcode.models.modules.dgl.gcn import GCNNet
from robustcode.models.modules.dgl.ggnn import GGNNNet
from robustcode.models.modules.dgl.iterators import EdgesFieldsGenerator
from robustcode.models.modules.dgl.iterators import GraphBatchIterator
from robustcode.models.modules.dgl.utransformer import UGraphTransformer
from robustcode.models.modules.iterators import SequenceIterator
from robustcode.models.modules.rnn.deeptyper import DeepTyper
from robustcode.models.modules.rnn.deeptyper import DeepTyperEdgesFieldsGenerator
from robustcode.models.modules.rnn.model import RNN
from robustcode.models.modules.rnn.model import RNNWithAttention
from robustcode.util.misc import boolean_string
from robustcode.util.misc import Logger


def tree_to_data(tree: AstTree, sample):
    if tree.analyzer is None:
        tree.analyzer = TypeScriptGraphAnalyzer()

    """
    Recompute the nodes since the tree structure was changed by modifications.    
    """
    tree.nodes = list(tree.root.forEachNode())

    types = [node.fields.get("types", node.type) for node in tree.nodes]
    values = [node.fields.get("values", node.value) for node in tree.nodes]
    depth = [node.depth() for node in tree.nodes]
    target = [
        node.fields.get("target", sample.target[node.id] if node.id >= 0 else "<null>")
        for node in tree.nodes
    ]
    mask_valid = [
        node.fields.get("mask_valid", sample.mask_valid[node.id] if node.id >= 0 else 0)
        for node in tree.nodes
    ]

    data = {
        "id": "{}_mod".format(sample.id),
        "values": values,
        "types": types,
        "target": target,
        "mask_valid": mask_valid,
        "depth": depth,
        "order": sample.order,
    }

    tree.number_nodes()
    per_type_edges = tree.compute_all_edges()
    for edge_type, values in per_type_edges.items():
        data[edge_type + "_src"] = [v[0] for v in values]
        data[edge_type + "_tgt"] = [v[1] for v in values]
    return data


class Dataset:
    @staticmethod
    def args():
        parser = argparse.ArgumentParser("DeepTyper AST Dataset", add_help=False)
        parser.add_argument(
            "--dataset",
            type=str,
            default="deeptyperast_4k",
            help="dataset name",
            choices=Config.DATASETS.keys(),
        )
        parser.add_argument(
            "--dataset_test", type=str, help="test dataset name (if any)"
        )

        parser.add_argument(
            "--cache_dir",
            type=str,
            default="../../../data",
            help="Directory with cached datasets",
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            default="../../datasets/types/data/",
            help="Path to raw dataset files",
        )
        parser.add_argument(
            "--num_samples", type=int, default=-1, help="number of samples to use"
        )
        return parser

    def tree_to_sample(self, tree_id, tree: AstTree):
        data = tree_to_data(tree, self.get_sample_by_id(tree_id))

        sample = torchtext.data.Example.fromlist(
            [data[name] for name, _ in self.fields], self.fields
        )

        self.id_to_sample[self.ID.vocab.stoi[sample.id]] = sample
        return sample

    def __init__(self, args, include_edges=False, dataset_eval=None):
        Logger.debug("Dataset: {}".format(args.dataset))
        config = Config.get_dataset(args.dataset)
        config.init(args.dataset_path, args.cache_dir)

        self.EDGES = (
            [
                "child_edges",
                # 'computed_from_edges',
                # 'next_token_edges',
                "returns_to_edges",
                "last_write_edges",
                "last_read_edges",
                "last_lexical_usage_edges",
            ]
            if include_edges
            else []
        )

        # Type of the AST node (e.g., Identifier, BinaryExpression, IfStatement, etc.)
        self.TYPES = torchtext.data.Field(sequential=True, include_lengths=True)
        # Value of the AST node (e.g., x, y, 5, console, etc.)
        self.VALUES = torchtext.data.Field(sequential=True)

        # Target to predict, in our case type of the extession (e.g., int, string, string[], etc.)
        # torchtext 0.4 has hardcoded values of unk_token='<unk>', any other value will break the batching
        self.TARGET = torchtext.data.Field(
            sequential=True, unk_token="<unk>", is_target=True
        )
        # User provided type annotation
        self.GOLD = torchtext.data.Field(sequential=True, unk_token="<unk>")

        # ID of the file from which the sample was generated
        self.ID = torchtext.data.Field(sequential=False, use_vocab=True)
        # order of the sample. Used to keep batching deterministic
        self.ORDER = torchtext.data.Field(sequential=False, use_vocab=False)

        # boolean mask that denotes for which AST nodes the prediction should be made.
        # In our case, these are nodes that have a type (e.g, all identifiers and expressions)
        self.MASK_VALID = torchtext.data.Field(
            sequential=True, use_vocab=False, pad_token=0, dtype=torch.uint8
        )
        self.MASK_GOLD = torchtext.data.Field(
            sequential=True, use_vocab=False, pad_token=0, dtype=torch.uint8
        )

        # depth of the node in the AST (i.e., distance from the root)
        self.DEPTH = torchtext.data.Field(
            sequential=True, use_vocab=False, pad_token=0, dtype=torch.uint8
        )
        # position of the node with respect to the parent in the AST.
        # 0 denotes the first child, 1 is the second child, etc.
        # self.POS = torchtext.data.Field(sequential=True, pad_token=0)

        # files used by TypeScript to infer the ground-truth types
        # useful in case modifications are applied to the file and type inference needs to be executed again
        # self.DEPENDENCIES = torchtext.data.Field(sequential=True)

        fields = {
            "ast_types": ("types", self.TYPES),
            "ast_values": ("values", self.VALUES),
            "id": ("id", self.ID),
            # 'pos': ('pos', self.POS),
            "target_full": ("target", self.TARGET),
            "mask_valid_full": ("mask_valid", self.MASK_VALID),
            # 'gold_type': ('gold', self.GOLD),
            # 'mask_gold': ('mask_gold', self.MASK_GOLD),
            "depth": ("depth", self.DEPTH),
            # 'dependencies': ('dependencies', self.DEPENDENCIES)
        }

        for edge_type in self.EDGES:
            fields[edge_type + "_src"] = (
                edge_type + "_src",
                torchtext.data.Field(
                    sequential=True,
                    pad_token=-1,
                    include_lengths=True,
                    use_vocab=False,
                    dtype=torch.int16,
                ),
            )
            fields[edge_type + "_tgt"] = (
                edge_type + "_tgt",
                torchtext.data.Field(
                    sequential=True, pad_token=-1, use_vocab=False, dtype=torch.int16
                ),
            )

        Logger.start_scope("Reading Dataset")
        dataset_cls = (
            torchtext.data.TabularDataset
            if not config.compressed
            else TabularCompressedDataset
        )
        dtrain, dvalid, dtest = dataset_cls.splits(
            path=os.path.join(args.cache_dir, args.dataset),
            train=config.train,
            validation=config.valid,
            test=config.test,
            format="json",
            fields=fields,
        )

        if args.num_samples != -1:
            dtrain.examples = dtrain.examples[: args.num_samples]
            dvalid.examples = dvalid.examples[: args.num_samples]

        if dataset_eval is not None:
            eval_ids = set(
                sample.id
                for sample in itertools.chain(dataset_eval.dvalid, dataset_eval.dtest)
            )
            dvalid.examples = [sample for sample in dvalid if sample.id in eval_ids]
            dtest.examples = [sample for sample in dtest if sample.id in eval_ids]

        self.filter_size(dtrain)
        self.filter_size(dvalid)
        self.filter_size(dtest)

        types = [
            "string",
            "number",
            "boolean",
            "void",
            "() => string",
            "() => number",
            "() => boolean",
            "() => void",
            "<null>",
        ]

        self.__normalize_values(itertools.chain(dtrain, dvalid, dtest))

        Logger.end_scope()
        Logger.start_scope("Processing Dataset")
        self.TYPES.build_vocab(dtrain)
        self.VALUES.build_vocab(dtrain, min_freq=10)
        self.TARGET.build_vocab(dtrain, max_size=0)
        self.TARGET.vocab.extend(SimpleNamespace(itos=types))
        # special value denoting no prediction
        self.TARGET.vocab.extend(SimpleNamespace(itos=["reject", "unsound"]))

        # use to replace types/values with predictions
        # make the types unique such that they are not mixed with the original values
        ext_types = ["__<" + v + ">__" for v in types + ["<unk>"]]
        self.TYPES.vocab.extend(SimpleNamespace(itos=ext_types))
        self.VALUES.vocab.extend(SimpleNamespace(itos=ext_types))

        # values >= than fixed_value_offset are replaced manually to denote predictions from previous iterations
        self.fixed_value_offset = self.VALUES.vocab.stoi[ext_types[0]]

        self.GOLD.build_vocab(dtrain, max_size=0)
        self.GOLD.vocab.extend(SimpleNamespace(itos=types))
        # self.POS.build_vocab(dtrain)

        self.__remove_null_labels(itertools.chain(dtrain, dvalid, dtest))

        # ID is for debugging so we build vocab also from test dataset
        # self.ID.build_vocab(dtrain, dvalid)
        self.ID.build_vocab(
            [sample.id for sample in itertools.chain(dtrain, dvalid, dtest)]
            + [
                "{}_mod".format(sample.id)
                for sample in itertools.chain(dtrain, dvalid, dtest)
            ]  # modified adversarial samples
        )
        # self.DEPENDENCIES.build_vocab(dtrain, dvalid)
        Logger.debug(
            " TYPES Vocab Size: {:6d}/{:6d}".format(
                len(self.TYPES.vocab), len(self.TYPES.vocab.freqs)
            )
        )
        Logger.debug(
            "VALUES Vocab Size: {:6d}/{:6d}".format(
                len(self.VALUES.vocab), len(self.VALUES.vocab.freqs)
            )
        )

        Logger.debug(
            "TARGET Vocab Size: {:6d}/{:6d}".format(
                len(self.TARGET.vocab), len(self.TARGET.vocab.freqs)
            )
        )
        for key, value in self.TARGET.vocab.freqs.most_common(20):
            print(
                "\t{:10s} {:10d}: {}".format(
                    "[vocab]" if key in self.TARGET.vocab.stoi else "", value, key
                )
            )

        Logger.end_scope()

        # Store the results
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.dtest = dtest

        self.pad_token_id = self.VALUES.vocab.stoi[self.VALUES.pad_token]
        self.unk_token_id = self.VALUES.vocab.stoi[self.VALUES.unk_token]
        self.reject_token_id = self.TARGET.vocab.stoi["reject"]
        self.unsound_token_id = self.TARGET.vocab.stoi["unsound"]

        self.id_to_sample = {
            self.ID.vocab.stoi[sample.id]: sample
            for sample in itertools.chain(self.dtrain, self.dvalid, self.dtest)
        }

        self.__init_order(self.dtrain)
        self.__init_order(self.dvalid)
        self.__init_order(self.dtest)

        self.fields = [(name, field) for name, field in dtrain.fields.items()]

    def __init_order(self, dataset):
        """
        Compute Batch Order
        We keep the order of samples in a batch fixed is it allows simpler implementation of the adversarial models

        The batch order we used is to sort the samples by the number of valid masks
        This makes the gradient computation fast (since it gradients are computed only for valid masks)
        and also batches together samples of roughly the same length
        The order is precomputed here since the number of valid masks can change during the training
        """

        sample_mask_size = [
            (sum(sample.mask_valid), idx, sample) for idx, sample in enumerate(dataset)
        ]
        sample_mask_size.sort()

        dataset.fields["order"] = self.ORDER
        for idx, (_, _, sample) in enumerate(sample_mask_size):
            setattr(sample, "order", idx)

    def __normalize_values(self, samples):
        for sample in samples:
            for node_id, target in enumerate(sample.target):
                if target in ["true", "false"]:
                    sample.target[node_id] = "boolean"
                if target == "String":
                    sample.target[node_id] = "string"

    def __remove_null_labels(self, samples):
        for sample in samples:
            for idx, target in enumerate(sample.target):

                if target == "<null>":
                    sample.mask_valid[idx] = 0

    def find_batch_for_sample(self, it, sample):
        idx = self.ID.vocab.stoi[sample.id]
        for batch in it:
            if idx is not None and idx not in batch.ids:
                continue
            return batch, numpy.flatnonzero(batch.ids == idx)[0]
        assert False

    def samples_for_batch(self, batch):
        ids = batch.ids if hasattr(batch, "ids") else batch.id
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        return [self.id_to_sample[idx] for idx in ids]

    def get_sample_by_id(self, idx):
        assert idx in self.id_to_sample
        return self.id_to_sample[idx]

    def sample_id(self, sample):
        return self.ID.vocab.stoi[sample.id]

    def filter_size(self, dataset, min_size=100):
        original_size = len(dataset)
        dataset.examples = [
            example for example in dataset if len(example.target) > min_size
        ]
        Logger.debug("Filter size {} -> {}".format(original_size, len(dataset)))

    def remove_duplicates(self, reverse=False):
        datasets = [self.dtrain, self.dvalid, self.dtest]
        if reverse:
            datasets.reverse()
        sim_dataset = fastsim.Dataset(
            [sample.id for sample in itertools.chain(*datasets)],
            [
                fastsim.hash_file_tokens(sample.values, ngram=2)
                for sample in itertools.chain(*datasets)
            ],
        )
        removed_ids, removed_pairs = fastsim.compute_similar_files(
            sim_dataset, num_perm=32, seed=42, threshold=0.7
        )

        def filter_samples(blacklist, data, name):
            original_size = len(data)
            data.examples = [example for example in data if example.id not in blacklist]
            Logger.debug("{} size {} -> {}".format(name, original_size, len(data)))

        filter_samples(removed_ids, self.dtrain, "Train")
        filter_samples(removed_ids, self.dvalid, "Valid")
        filter_samples(removed_ids, self.dtest, "Test")


class Models(Enum):
    RNN = RNN
    RNNAttn = RNNWithAttention
    UGraphTransformer = UGraphTransformer
    GCN = GCNNet
    GGNN = GGNNNet
    DeepTyper = DeepTyper

    @staticmethod
    def args():
        parser = argparse.ArgumentParser("Models", add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Model Type",
            choices=[v.name for v in Models],
        )
        parser.add_argument("--use_cuda", type=boolean_string, default=True)
        parser.add_argument("--include_values", type=boolean_string, default=True)
        return parser

    @staticmethod
    def make(args, dataset: Dataset, device, train_iter):
        num_edge_types = (
            train_iter.edge_gen.num_edge_types()
            if isinstance(train_iter, GraphBatchIterator)
            else None
        )
        for type in Models:
            if type.name == args.model:
                model = type.value(
                    args,
                    [len(dataset.TYPES.vocab), len(dataset.VALUES.vocab)]
                    if args.include_values
                    else [len(dataset.TYPES.vocab)],
                    len(dataset.TARGET.vocab),
                    device=device,
                    pad_token_id=dataset.pad_token_id,
                    num_edge_types=num_edge_types,
                )
                if torch.cuda.is_available() and args.use_cuda:
                    model = model.cuda(device=device)
                return model
        assert False


class Iterators:
    @staticmethod
    def args():
        parser = argparse.ArgumentParser("Iterators", add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--bptt_len", type=int)
        parser.add_argument("--window_size", type=int, default=0)
        parser.add_argument(
            "--per_node_target",
            type=boolean_string,
            default=True,
            help="Whether and output is predicted for each node (e.g., for type inference) or once per whole graph (e.g., code captioning)",
        )
        return parser

    @staticmethod
    def make_single(
        args,
        model_type,
        device,
        masks,
        edges: List[str],
        dataset: torchtext.data.Dataset,
        train=True,
        cached=False,
    ):
        input_fields = (
            ["types", "values"]
            if not hasattr(args, "include_values") or args.include_values
            else ["types"]
        )
        if model_type in [Models.RNN, Models.RNNAttn]:
            diter = torchtext.data.BucketIterator(
                dataset=dataset,
                batch_size=args.batch_size,
                device=device,
                sort_key=lambda x: x.order,
                shuffle=True,
                sort_within_batch=False,
                sort=True,
                repeat=False,
                train=train,
            )
            diter = SequenceIterator(
                it=diter,
                input_fields=input_fields,
                target_field="target",
                mask_fields=masks.keys(),
            )
        elif model_type in [
            Models.UGraphTransformer,
            Models.GCN,
            Models.GGNN,
            Models.DeepTyper,
        ]:
            if model_type == Models.DeepTyper:
                edge_gen = DeepTyperEdgesFieldsGenerator(device)
            else:
                edge_gen = EdgesFieldsGenerator(
                    edges, device=device, self_loop=True, window_size=args.window_size
                )
            diter = GraphBatchIterator(
                dataset,
                input_fields=input_fields,
                target_field="target",
                mask_fields=masks.keys(),
                batch_size=args.batch_size,
                device=device,
                sort_key=lambda x: x.order,
                edge_gen=edge_gen,
                cached=cached,
                per_node_target=args.per_node_target,
            )
        else:
            assert False, "Unhandled Model Type {}".format(model_type)

        return diter

    @staticmethod
    def make(args, model_type, dataset: Dataset, device, masks):
        train_iter = Iterators.make_single(
            args,
            model_type,
            device,
            masks,
            dataset.EDGES,
            dataset.dtrain,
            train=True,
            cached=True,
        )
        valid_iter = Iterators.make_single(
            args,
            model_type,
            device,
            masks,
            dataset.EDGES,
            dataset.dvalid,
            train=False,
            cached=True,
        )
        test_iter = Iterators.make_single(
            args,
            model_type,
            device,
            masks,
            dataset.EDGES,
            dataset.dtest,
            train=False,
            cached=False,
        )
        return train_iter, valid_iter, test_iter
