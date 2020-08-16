import sys
from typing import Dict

import torchtext

from robustcode.analysis.graph import AstTree
from robustcode.util.misc import Logger


def batch_to_trees(batch):
    batch_types, lengths = batch.types
    lengths = lengths.cpu().numpy()
    batch_types = batch_types.cpu().transpose(0, 1).numpy()
    batch_values = batch.values.cpu().transpose(0, 1).numpy()
    batch_depths = batch.depth.cpu().transpose(0, 1).numpy()

    trees = []
    for length, types, values, depths in zip(
        lengths, batch_types, batch_values, batch_depths
    ):
        tree = AstTree.fromTensor(types[:length], values[:length], depths[:length],)
        trees.append(tree)
    return trees


def iter_to_trees(iter) -> Dict[int, AstTree]:
    Logger.start_scope("Converting Iter to Trees")
    trees = {}
    for batch in iter:
        batch_trees = batch_to_trees(batch)
        for tree, idx in zip(batch_trees, batch.id):
            trees[idx.item()] = tree
        sys.stderr.write("\r{}".format(len(trees)))

    sys.stderr.write("\r")
    Logger.debug("# Trees: {}".format(len(trees)))
    Logger.end_scope()
    return trees


def dataset_to_trees(dataset, ID, analyzer=None) -> Dict[int, AstTree]:
    Logger.start_scope("Converting Dataset to Trees")
    trees = {}
    for sample in dataset:
        tree = AstTree.fromTensor(
            sample.types, sample.values, sample.depth, {"target": sample.target}
        )
        tree.analyzer = analyzer
        trees[ID.vocab.stoi[sample.id]] = tree
        sys.stderr.write("\r{}".format(len(trees)))

    sys.stderr.write("\r")
    Logger.debug("# Trees: {}".format(len(trees)))
    Logger.end_scope()
    return trees


def dataset_to_trees_num(dataset):
    it = torchtext.data.BucketIterator(
        dataset=dataset,
        batch_size=1,
        sort_key=lambda x: len(x.target),
        shuffle=False,
        sort_within_batch=False,
        repeat=False,
    )

    return iter_to_trees(it)
