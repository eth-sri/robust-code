import collections
import functools
import itertools
import os
import pickle
import sys
from types import SimpleNamespace
from typing import Iterable
from typing import List
from typing import Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import optim

from robustcode.analysis.graph import AstTree
from robustcode.models.modules.dgl.gcn import GCNNet
from robustcode.models.modules.dgl.ggnn import GGNNNet
from robustcode.models.modules.dgl.utransformer import GraphModel
from robustcode.models.modules.dgl.utransformer import UGraphTransformer
from robustcode.models.modules.iterators import MiniBatch
from robustcode.models.modules.neural_model_base import NeuralModelBase
from robustcode.models.modules.rejection_cross_entropy_loss import (
    RejectionCrossEntropyLoss,
)
from robustcode.models.modules.rnn.model import RNNWithAttention
from robustcode.models.modules.util import Random
from robustcode.models.robust.adversary.adversary import AdversarialMode
from robustcode.models.robust.adversary.adversary import AdversaryBase
from robustcode.models.robust.adversary.adversary import AdversaryBatchIter
from robustcode.models.robust.adversary.node_grads import each_node_grads
from robustcode.models.robust.dataset import Dataset
from robustcode.models.robust.dataset import Iterators
from robustcode.models.robust.dataset import Models
from robustcode.models.robust.gurobi.edge_optimizer import EdgeFilter
from robustcode.models.robust.gurobi.edge_optimizer import EdgeOptimizer
from robustcode.models.robust.gurobi.edge_optimizer import FilteredGraphIterator
from robustcode.models.robust.utils import checkpoint_dir
from robustcode.models.robust.utils import eval_adversarial
from robustcode.models.robust.utils import load_model
from robustcode.models.robust.utils import make_adversary
from robustcode.models.robust.utils import save_model
from robustcode.models.robust.utils import train_base_model
from robustcode.models.robust.visualization import BooleanWithMaskLabel
from robustcode.models.robust.visualization import FieldLabel
from robustcode.models.robust.visualization import TreeVisualization
from robustcode.util.argparse import ArgConfigParser
from robustcode.util.misc import acc
from robustcode.util.misc import boolean_string
from robustcode.util.misc import Logger
from robustcode.util.misc import Timer


def parse_args():
    parser = ArgConfigParser(
        "Robust Code Sparse Training (with Abstain)",
        parents=[
            Random.args(),
            Dataset.args(),
            Models.args(),
            Iterators.args(),
            RNNWithAttention.args(),
            GraphModel.args(),
            UGraphTransformer.args(),
            GCNNet.args(),
            GGNNNet.args(),
            RejectionCrossEntropyLoss.args(),
        ],
    )
    parser.add_argument("--use_cuda", type=boolean_string, default=True)
    parser.add_argument(
        "--out_file", type=str, default=None, help="file to save evaluation statistics"
    )
    parser.add_argument("--log_file", type=str, default=None, help="file to store logs")
    parser.add_argument(
        "--adversarial",
        type=boolean_string,
        default=True,
        help="Whether to use adversarial training",
    )
    parser.add_argument(
        "--adv_mode",
        type=AdversarialMode.from_string,
        choices=list(AdversarialMode),
        default=AdversarialMode.RANDOM,
    )
    parser.add_argument(
        "--n_renames",
        type=int,
        default=20,
        help="Number of iterations for adversarial renaming",
    )
    parser.add_argument(
        "--n_subtree",
        type=int,
        default=30,
        help="Number of iterations for adversarial subtree replacement",
    )
    parser.add_argument(
        "--train_adv_mode",
        type=AdversarialMode.from_string,
        choices=list(AdversarialMode),
        default=AdversarialMode.RANDOM,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times the experiment is repeated with a different seed. Used to compute variance.",
    )
    parser.add_argument(
        "--max_models", type=int, default=1, help="Number of robust models to train"
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="Whether to run only the evaluation",
    )
    parser.add_argument(
        "--min_nonabstained",
        type=int,
        default=500,
        help="Minimum number of samples for which each model should not abstain",
    )
    parser.add_argument("--last_base", default=False, action="store_true")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory to save checkouts and results",
    )
    parser.add_argument("--tag", type=str, default="default", help="Experiment tag")
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of training epochs"
    )
    args = parser.parse_args()
    return args


def compute_edge_filter(
    it, model, dataset, loss_function, threshold=0.5, verbose=False
):
    timers = collections.defaultdict(Timer)
    edge_optimizer = EdgeOptimizer()
    for node_grads in each_node_grads(
        it, model, dataset, loss_function, threshold=threshold, max_samples=30
    ):
        if torch.any(node_grads.probs > 0.1):
            tgt_nodes = (
                torch.masked_select(node_grads.tgt_nodes, node_grads.probs > 0.1)
                .cpu()
                .numpy()
            )
            probs = (
                torch.masked_select(node_grads.probs, node_grads.probs > 0.1)
                .cpu()
                .numpy()
            )
        else:
            tgt_nodes = node_grads.tgt_nodes[:3].cpu().numpy()
            probs = node_grads.probs[:3].cpu().numpy()

        if len(tgt_nodes) == 0:
            Logger.debug(
                "Empty target nodes: src: {}, tgt_nodes: {}, {}".format(
                    node_grads.src_node, node_grads.tgt_nodes, node_grads.probs
                )
            )
            continue

        debug_info = "" if verbose else None

        timers["nodes"].start()
        depth = max(
            nx.shortest_path_length(
                node_grads.rev_tree_nx, source=node_grads.src_node, target=tgt
            )
            for tgt in tgt_nodes
        )
        nodes = [node_grads.src_node] + list(
            itertools.chain.from_iterable(
                successors
                for _, successors in nx.bfs_successors(
                    node_grads.rev_tree_nx,
                    source=node_grads.src_node,
                    depth_limit=depth,
                )
            )
        )
        assert all(tgt_node in nodes for tgt_node in tgt_nodes)
        if verbose:
            debug_info += "nodes: {}\n".format(nodes)
        timers["nodes"].stop()

        timers["edges"].start()
        edges = [
            (i, j)
            for (i, j) in node_grads.tree_nx.edges(nodes)
            if i in nodes and j in nodes
        ]
        if verbose:
            debug_info += "edges: {}\n".format(edges)
        timers["edges"].stop()

        timers["arcs"].start()
        features = EdgeFilter.edge_features(
            edges, node_grads.tree, debug_info=debug_info
        )
        arcs = {}
        for (i, j), feature in zip(edges, features):
            if i == j:
                # split self-loops into new nodes
                # Needed when using self-loops as otherwise the same node both generates and consumes inflow
                i = "{}r".format(i)
            arcs[(str(i), str(j))] = feature  # '{}_{}'.format(node_type, edge_type)
        if verbose:
            debug_info += "arcs: {}\n".format(arcs)
            debug_info += "features: {}\n".format(features)
        timers["arcs"].stop()

        # update list of notes with newly generated ones
        nodes = set()
        for (i, j) in arcs.keys():
            nodes.add(i)
            nodes.add(j)
        if verbose:
            debug_info += "nodes: {}\n".format(nodes)

        tgt_nodes = [
            str(v) if v != node_grads.src_node else "{}r".format(v) for v in tgt_nodes
        ]
        inflow = {
            tgt_node: int(p * 100)
            for tgt_node, p in zip(tgt_nodes, probs)
            if tgt_node in nodes
        }
        inflow[str(node_grads.src_node)] = -sum(inflow.values())
        if verbose:
            debug_info += "inflow: {}\n".format(inflow)
        if len(arcs) == 0:
            continue
        edge_optimizer.add_sample(nodes, arcs, inflow)

        if verbose:
            edge_optimizer_tmp = EdgeOptimizer()
            edge_optimizer_tmp.add_sample(nodes, arcs, inflow)
            edge_optimizer_tmp.solve(debug_info=debug_info)

    for key, timer in timers.items():
        Logger.debug("{}: {}".format(key, timer))
    edge_filter = edge_optimizer.solve()
    edge_filter.print(dataset=dataset)  # , edge_gen=it.edge_gen)
    return edge_filter


def mask_count(it, mask_field="mask_valid"):
    return sum(torch.sum(batch.masks[mask_field]).item() for batch in it)


def number_of_edges(it):
    return number_of_nodes_and_edges(it)[1]


def number_of_nodes_and_edges(it):
    num_edges = 0
    num_nodes = 0
    for batch in it:
        g = batch.X
        num_edges += g.number_of_edges()
        num_nodes += g.number_of_nodes()
    return num_nodes, num_edges


def get_rejection_thresholds(
    it, model: NeuralModelBase, dataset: Dataset, precision_thresholds: Iterable[float]
):
    num_bins = 1000
    # stats = [SimpleNamespace(correct=0, total=0) for _ in range(num_bins + 1)]

    num_correct = torch.zeros(num_bins)
    num_total = torch.zeros(num_bins)
    for batch in tqdm.tqdm(it, ncols=100, leave=False):
        _, best_predictions, reject_probs = model.predict_probs_with_reject(
            batch, reject_id=dataset.reject_token_id
        )
        mask = model.padding_mask(batch, mask_field="mask_valid")
        targets = batch.Y

        best_predictions = best_predictions.masked_select(mask)
        reject_probs = reject_probs.masked_select(mask).cpu()
        targets = targets.masked_select(mask)

        is_corrects = (targets == best_predictions).cpu()

        num_total.add_(torch.histc(reject_probs, bins=num_bins, min=0, max=1))
        num_correct.add_(
            torch.histc(
                reject_probs.masked_select(is_corrects), bins=num_bins, min=0, max=1
            )
        )

    def precision(stat):
        if stat.total == 0:
            return 0
        return stat.correct * 1.0 / stat.total

    thresholds = [SimpleNamespace(h=None, size=0) for _ in precision_thresholds]
    rolling_stat = SimpleNamespace(correct=0, total=0)
    for i, correct, total in zip(
        itertools.count(), num_correct.numpy(), num_total.numpy()
    ):
        for t, precision_threshold in zip(thresholds, precision_thresholds):
            if precision_threshold <= precision(rolling_stat):
                # update threshold if it's not set or the number of samples increased
                if t.h is None or t.size * 1.01 < rolling_stat.total:
                    t.h = i / float(num_bins)
                    t.size = int(rolling_stat.total)

        rolling_stat.correct += correct
        rolling_stat.total += total

    Logger.debug(
        "Thresholds: {}, sizes: {}".format(
            [t.h for t in thresholds], [t.size for t in thresholds]
        )
    )
    return thresholds


def print_rejection_thresholds(it, model: NeuralModelBase, dataset: Dataset):
    num_correct = 0
    num_total = 0
    thresholds = np.arange(0.1, 1.1, 0.1)
    stats = collections.defaultdict(lambda: SimpleNamespace(correct=0, total=0))
    for batch in tqdm.tqdm(it, ncols=100, leave=False):
        _, best_predictions, reject_probs = model.predict_probs_with_reject(
            batch, reject_id=dataset.reject_token_id
        )
        mask = model.padding_mask(batch, mask_field="mask_valid")
        targets = batch.Y

        best_predictions = best_predictions.masked_select(mask)
        reject_probs = reject_probs.masked_select(mask)
        targets = targets.masked_select(mask)

        is_correct = targets == best_predictions
        num_correct += torch.sum(is_correct).item()
        num_total += targets.numel()

        for h in thresholds:
            h_mask = reject_probs <= h
            stats[h].total += torch.sum(h_mask).item()
            stats[h].correct += torch.sum(is_correct.masked_select(h_mask)).item()

    for h in thresholds:
        Logger.debug(
            "Threshold {:5.2f}: {:6d}/{:6d} ({:.2f}%)".format(
                h,
                stats[h].correct,
                stats[h].total,
                acc(stats[h].correct, stats[h].total),
            )
        )

    Logger.debug(
        "{:6d}/{:6d} ({:.2f}%)".format(
            num_correct, num_total, acc(num_correct, num_total)
        )
    )


def train_model(
    model: NeuralModelBase,
    dataset: Dataset,
    num_epochs,
    train_iter,
    valid_iter,
    lr=0.001,
    weight=None,
    target_o=1.0,
):
    # model.reset_parameters()
    opt = optim.Adam(model.parameters(), lr=lr)
    Logger.start_scope("Training Model")

    o_base = len(dataset.TARGET.vocab) - 4  # 'reject', '<unk>', '<pad>'
    loss_function = RejectionCrossEntropyLoss(
        o_base,
        len(dataset.TARGET.vocab),
        dataset.reject_token_id,
        reduction="none",
        weight=weight,
    )
    model.loss_function = loss_function
    model.opt = opt

    step = 1.0 / (num_epochs // 2)
    schedule = [
        f * o_base + (1 - f) * 1.0 for f in np.arange(start=1.0, stop=0.0, step=-step)
    ]
    schedule += [
        f * ((1.0 + schedule[-1]) / 2) + (1 - f) * target_o
        for f in np.arange(start=1.0, stop=0.0, step=-step)
    ]
    schedule += [target_o] * (num_epochs // 2)

    train_prec, valid_prec = None, None
    for epoch, o_upper in enumerate(schedule):
        Logger.start_scope("Epoch {}, o_upper={:.3f}".format(epoch, o_upper))
        loss_function.o = o_upper
        model.fit(train_iter, opt, loss_function, mask_field="mask_valid")

        valid_stats = model.accuracy(
            valid_iter, dataset.TARGET
        )  # , thresholds=[0.5, 0.8, 0.9, 0.95])
        valid_prec = valid_stats["mask_valid_noreject_acc"]
        Logger.debug(f"valid_prec: {valid_prec}")
        Logger.end_scope()

        # Logger.start_scope('Print Rejection Thresholds')
        # print_rejection_thresholds(train_iter, model, dataset)
        # print_rejection_thresholds(valid_iter, model, dataset)
        # Logger.end_scope()

        # Logger.start_scope('Get Rejection Thresholds')
        # get_rejection_thresholds(train_iter, model, dataset, [1.00, 0.99, 0.95, 0.9, 0.8])
        # get_rejection_thresholds(valid_iter, model, dataset, [1.00, 0.99, 0.95, 0.9, 0.8])
        # Logger.end_scope()

    train_stats = model.accuracy(train_iter, dataset.TARGET, verbose=False)
    train_prec = train_stats["mask_valid_noreject_acc"]
    Logger.debug(f"train_prec: {train_prec}, valid_prec: {valid_prec}")
    Logger.end_scope()
    # exit(0)
    return train_prec, valid_prec


def train_nonempty_model(
    model_fn,
    dataset: Dataset,
    train_iter,
    valid_iter,
    num_epochs=10,
    max_steps=10,
    step=0.1,
):
    model = model_fn()
    train_model(model, dataset, num_epochs, train_iter, valid_iter, target_o=1.1)

    thresholds = get_rejection_thresholds(
        valid_iter, model, dataset, [0.98, 0.95, 0.9, 0.8]
    )
    thresholds = [t for t in thresholds if t.h is not None and t.size > 100]
    if not thresholds:
        thresholds = get_rejection_thresholds(
            train_iter, model, dataset, [0.98, 0.95, 0.9, 0.8]
        )
        thresholds = [t for t in thresholds if t.h is not None and t.size > 100]

    if thresholds:
        Logger.debug("Rejection Threshold: {}".format(thresholds[0]))
        model.accuracy_with_reject(
            valid_iter, dataset.TARGET, dataset.reject_token_id, thresholds[0].h
        )
        return model, thresholds[0]

    return None, None


class RobustModel:
    def __init__(
        self,
        model_fn,
        dataset,
        idx=None,
        rename_adversary: AdversaryBase = None,
        subtree_adversary: AdversaryBase = None,
        base_model=False,
    ):
        self.model: NeuralModelBase = model_fn()
        self.model_fn = model_fn
        self.base_model = base_model
        """
        rejects the prediction x if the p('reject' | x) >= threshold
        """
        self.threshold = 0.5
        self.dataset = dataset
        self.edge_filter: EdgeFilter = None
        self.idx = idx

        self.valid_stats = None

        self.rename_adversary = rename_adversary
        self.subtree_adversary = subtree_adversary

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.edge_filter = pickle.loads(state["edge_filter"])
        if self.edge_filter is not None:
            self.edge_filter.init_cache()

        self.threshold = state["threshold"]
        self.model.loss_function = RejectionCrossEntropyLoss(
            1,
            len(self.dataset.TARGET.vocab),
            self.dataset.reject_token_id,
            reduction="none",
        )

    def state_dict(self):
        if self.edge_filter is not None:
            self.edge_filter.clear_cache()
        state = {
            "model": self.model.state_dict(),
            "threshold": self.threshold,
            "edge_filter": pickle.dumps(self.edge_filter),
        }
        if self.edge_filter is not None:
            self.edge_filter.init_cache()
        return state

    def __model_name(self, model_id):
        name = "{}_{}".format(model_id, self.idx)
        if self.base_model:
            name += "_base"
        return name

    def load(self, args, model_id):
        return load_model(self, args, self.__model_name(model_id))

    def save(self, args, model_id):
        save_model(self, args, self.__model_name(model_id))

    def train(
        self,
        train_iter,
        valid_iter,
        num_epochs=10,
        min_nonabstained=500,
        depth=None,
        test_iter=None,
        apply_model=True,  # base_model=False,
        train_adv_mode=AdversarialMode.RANDOM,
        loaded=False,
        model_eval=None,
    ):
        if not loaded:
            if self.base_model:

                train_base_model(
                    self.model,
                    self.dataset,
                    10,
                    RobustModelBatchIter(
                        model_eval,
                        AdversaryBatchIter(
                            self.subtree_adversary,
                            self.model,
                            AdversaryBatchIter(
                                self.rename_adversary,
                                self.model,
                                train_iter,
                                num_samples=1,
                                adv_mode=train_adv_mode,
                            ),
                        ),
                    ),
                    [valid_iter],
                    verbose=False,
                )

                success = True
            else:
                success = self.__train_inner(
                    train_iter,
                    valid_iter,
                    num_epochs=num_epochs,
                    train_adv_mode=train_adv_mode,
                    min_nonabstained=min_nonabstained,
                )

            if not success:
                Logger.debug("model train failed")
                input()
                return False

        # if self.idx is not None:
        #     torch.save(self.state_dict(), '{:03d}_model.pt'.format(self.idx))

        # we cannot reuse the iterators from calling __train_inner as these are shuffled
        if self.edge_filter is not None:
            Logger.debug("Original Edges: #{}".format(number_of_edges(train_iter)))
            f_train_iter = FilteredGraphIterator.from_iter(train_iter, self.edge_filter)
            Logger.debug("Filtered Edges: #{}".format(number_of_edges(f_train_iter)))
            f_valid_iter = FilteredGraphIterator.from_iter(valid_iter, self.edge_filter)
            if test_iter is not None:
                f_test_iter = FilteredGraphIterator.from_iter(
                    test_iter, self.edge_filter
                )
        else:
            f_train_iter = train_iter
            f_valid_iter = valid_iter
            if test_iter is not None:
                f_test_iter = test_iter

        Logger.debug("Train Accuracy:")
        train_stats = self.model.accuracy_with_reject(
            f_train_iter,
            self.dataset.TARGET,
            self.dataset.reject_token_id,
            self.threshold,
        )
        Logger.debug("Valid Accuracy:")
        self.valid_stats = self.model.accuracy_with_reject(
            f_valid_iter,
            self.dataset.TARGET,
            self.dataset.reject_token_id,
            self.threshold,
        )
        if test_iter is not None:
            Logger.debug("Test Accuracy:")
            self.model.accuracy_with_reject(
                f_test_iter,
                self.dataset.TARGET,
                self.dataset.reject_token_id,
                self.threshold,
            )

        train_prec = train_stats["mask_valid_noreject_acc"]
        valid_prec = self.valid_stats["mask_valid_noreject_acc"]
        Logger.debug(f"train_prec: {train_prec}, valid_prec: {valid_prec}")

        if apply_model:
            self.apply(train_iter, f_train_iter, is_train=True)
            self.apply(valid_iter, f_valid_iter, is_train=True)
            if test_iter is not None:
                self.apply(test_iter, f_test_iter)
        return True

    def __refine_model_adversarial(
        self,
        model,
        train_iter,
        valid_iter,
        adv_train_iter,
        threshold,
        min_nonabstained=500,
    ):
        Logger.debug("fit_adversarial")
        step = 1.0 / 4
        schedule = [
            f * 1.1 + (1 - f) * 1.02 for f in np.arange(start=1.0, stop=0.0, step=-step)
        ] + 12 * [1.02]

        num_refined_all = []
        for epoch, o in enumerate(schedule):
            model.loss_function.o = o
            Logger.debug("Epoch {}, o={}".format(epoch, o))
            num_refined = self.subtree_adversary.fit_adversarial(
                model, train_iter, adv_train_iter, threshold.h
            )
            model.accuracy_with_reject(
                valid_iter,
                self.dataset.TARGET,
                self.dataset.reject_token_id,
                threshold.h,
            )

            # print_rejection_thresholds(valid_iter, model, self.dataset)

            thresholds = get_rejection_thresholds(
                valid_iter, model, self.dataset, [0.99, 0.98, 0.95, 0.9]
            )
            thresholds = [
                t
                for t in thresholds
                if t.h is not None and t.size > min_nonabstained * 2
            ]
            if not thresholds:
                return None
            threshold = thresholds[0]

            if num_refined == 0:
                break
            if epoch > 7 and num_refined * 3 >= sum(num_refined_all[-3:]):
                break
            num_refined_all.append(num_refined)

        thresholds = get_rejection_thresholds(
            valid_iter, model, self.dataset, [1.00, 0.99, 0.98, 0.95]
        )
        thresholds = [
            t for t in thresholds if t.h is not None and t.size > min_nonabstained
        ]
        Logger.debug("Selected Threshold: {}".format(thresholds))
        assert thresholds
        return thresholds[0]

    def __refine_model(self, model, train_iter, valid_iter, min_nonabstained=500):
        opt = optim.Adam(model.parameters(), lr=0.0001)
        model.opt = opt
        step = 1.0 / 6
        schedule = [
            f * 2 + (1 - f) * 1.0 for f in np.arange(start=1.0, stop=0.0, step=-step)
        ] + 4 * [1.0]
        for o in schedule:
            model.loss_function.o = o
            model.fit(train_iter, opt, model.loss_function, mask_field="mask_valid")

        thresholds = get_rejection_thresholds(
            valid_iter, model, self.dataset, [0.98, 0.95, 0.9, 0.8]
        )
        thresholds = [
            t for t in thresholds if t.h is not None and t.size > min_nonabstained
        ]

        if not thresholds:
            return None
        return thresholds[0]

    def __copy_model(self, model):
        new_model = self.model_fn()
        new_model.load_state_dict(model.state_dict())
        new_model.loss_function = model.loss_function
        return new_model

    def accuracy(self, valid_iter, adversarial=False):
        return self.__accuracy(
            self.model, self.threshold, valid_iter, adversarial=adversarial
        )

    def __accuracy(self, base_model, reject_threshold, valid_iter, adversarial=False):
        valid_stats = base_model.accuracy_with_reject(
            valid_iter,
            self.dataset.TARGET,
            self.dataset.reject_token_id,
            reject_threshold,
        )

        if adversarial:
            Logger.start_scope("adversarial accuracy")

            stats = self.rename_adversary.adversarial_accuracy(
                base_model,
                valid_iter,
                [
                    AdversaryBatchIter(
                        self.subtree_adversary,
                        base_model,
                        AdversaryBatchIter(
                            self.rename_adversary, base_model, valid_iter, num_samples=2
                        ),
                    ),
                    AdversaryBatchIter(
                        self.rename_adversary, base_model, valid_iter, num_samples=40
                    ),
                ],
                threshold=reject_threshold,
                approximate=True,
            )
            Logger.end_scope()
            return stats.is_sound()

        valid_prec = valid_stats["mask_valid_noreject_acc"]
        return valid_prec == 100.0

    def make_adversary_iter(
        self, it, model, adv_mode=AdversarialMode.RANDOM, num_samples=1
    ):
        return AdversaryBatchIter(
            self.subtree_adversary,
            model,
            AdversaryBatchIter(
                self.rename_adversary,
                model,
                it,
                num_samples=num_samples,
                adv_mode=adv_mode,
            ),
        )

    def make_rename_adversary_iter(
        self, it, model, adv_mode=AdversarialMode.RANDOM, num_samples=20
    ):
        return AdversaryBatchIter(
            self.rename_adversary, model, it, num_samples=num_samples, adv_mode=adv_mode
        )

    def __train_inner(
        self,
        train_iter,
        valid_iter,
        num_epochs=10,
        train_adv_mode=AdversarialMode.RANDOM,
        min_nonabstained=500,
    ):

        model, threshold = train_nonempty_model(
            self.model_fn, self.dataset, train_iter, valid_iter, num_epochs=num_epochs
        )

        if model is None:
            Logger.debug("Nonempty model failed!")
            return False

        best_model = model
        best_threshold = threshold
        best_edge_filter = None

        self.__accuracy(model, threshold.h, valid_iter, adversarial=False)

        edge_filter = compute_edge_filter(
            train_iter,
            best_model,
            self.dataset,
            best_model.loss_function,
            threshold=threshold.h,
            verbose=True,
        )

        while True:
            Logger.debug(
                "Model with #{} non-rejected predictions".format(threshold.size)
            )

            Logger.debug("Original Edges: #{}".format(number_of_edges(train_iter)))
            train_iter = FilteredGraphIterator.from_iter(train_iter, edge_filter)
            Logger.debug("Filtered Edges: #{}".format(number_of_edges(train_iter)))
            valid_iter = FilteredGraphIterator.from_iter(valid_iter, edge_filter)

            model = self.__copy_model(model)
            threshold = self.__refine_model(
                model, train_iter, valid_iter, min_nonabstained=min_nonabstained
            )
            if threshold is None:
                break

            threshold = self.__refine_model_adversarial(
                model,
                train_iter,
                valid_iter,
                [
                    self.make_rename_adversary_iter(
                        train_iter, model, train_adv_mode, num_samples=5
                    ),
                    self.make_adversary_iter(
                        train_iter, model, train_adv_mode, num_samples=5
                    ),
                ],
                threshold,
                min_nonabstained=min_nonabstained,
            )

            if threshold is None:
                break

            best_model = model
            best_threshold = threshold
            best_edge_filter = edge_filter

            edge_filter = compute_edge_filter(
                train_iter,
                best_model,
                self.dataset,
                best_model.loss_function,
                threshold=threshold.h,
                verbose=False,
            )
            Logger.debug(
                "new edges: {} ({}), old edges: {}".format(
                    len(edge_filter), len(edge_filter) * 1.04, len(best_edge_filter)
                )
            )
            if len(edge_filter) * 1.04 >= len(best_edge_filter):
                # self.accuracy(model, threshold.h, valid_iter, adversarial=True)
                break

        if best_edge_filter is None:
            Logger.debug("No Edge Filter, training base model adversarially")
            best_threshold = self.__refine_model_adversarial(
                best_model,
                train_iter,
                valid_iter,
                [
                    self.make_rename_adversary_iter(train_iter, model, train_adv_mode),
                    self.make_adversary_iter(train_iter, model, train_adv_mode),
                ],
                best_threshold,
                min_nonabstained=min_nonabstained,
            )

        Logger.debug("Train Accuracy:")
        train_stats = best_model.accuracy_with_reject(
            train_iter,
            self.dataset.TARGET,
            self.dataset.reject_token_id,
            best_threshold.h,
        )
        Logger.debug("Valid Accuracy:")
        valid_stats = best_model.accuracy_with_reject(
            valid_iter,
            self.dataset.TARGET,
            self.dataset.reject_token_id,
            best_threshold.h,
        )

        train_prec = train_stats["mask_valid_noreject_acc"]
        valid_prec = valid_stats["mask_valid_noreject_acc"]
        Logger.debug(f"train_prec: {train_prec}, valid_prec: {valid_prec}")

        self.edge_filter = best_edge_filter
        self.model = best_model
        self.threshold = best_threshold.h
        self.model.accuracy_with_reject(
            train_iter,
            self.dataset.TARGET,
            self.dataset.reject_token_id,
            self.threshold,
        )

        return True

    def predict_with_reject(self, batch: MiniBatch, apply=True):
        # assert self.edge_filter
        if self.edge_filter is not None:
            fbatch = FilteredGraphIterator.filter_batch(batch, self.edge_filter)
        else:
            fbatch = batch  # .clone()
        preds = self.model.predict_with_reject(
            fbatch, reject_id=self.dataset.reject_token_id, threshold=self.threshold
        )
        if apply:
            self.apply_batch(batch, fbatch, num_verbose=0, preds=preds)
        return preds

    def apply(
        self, it, filtered_it, mask_field="mask_valid", num_verbose=0, is_train=False
    ):
        num_predicted = 0
        num_shown = 0
        for batch, fbatch in zip(it, filtered_it):
            num_predicted_batch, num_shown_batch = self.apply_batch(
                batch,
                fbatch,
                mask_field=mask_field,
                num_verbose=max(0, num_verbose - num_shown),
                is_train=is_train,
            )
            num_predicted += num_predicted_batch
            num_shown += num_shown_batch

        Logger.debug("Number of predicted nodes: {}".format(num_predicted))

    def apply_batch(
        self,
        batch: MiniBatch,
        fbatch: MiniBatch,
        mask_field="mask_valid",
        num_verbose=0,
        preds=None,
        is_train=False,
    ):
        num_predicted = 0
        num_shown = 0

        if preds is None:
            preds = self.model.predict_with_reject(
                fbatch, self.dataset.reject_token_id, threshold=self.threshold
            )
        gts = fbatch.Y

        g = batch.X
        p_mask = preds != self.dataset.reject_token_id
        t_mask = self.model.padding_mask(batch, mask_field=mask_field)
        mask = (t_mask & p_mask).cpu()

        masked_nodes = g.nodes().masked_select(mask)
        masked_pred = preds.cpu().masked_select(mask)
        masked_ground_truth = gts.cpu().masked_select(mask)

        num_predicted += masked_nodes.numel()
        if masked_nodes.numel() == 0:
            return num_predicted, num_shown

        # replace types and values fields with the prediction
        # '__<' + v + '>__'
        if is_train:
            pred_words = [
                "__<" + self.dataset.TARGET.vocab.itos[p] + ">__"
                for p in masked_ground_truth
            ]
        else:
            pred_words = [
                "__<" + self.dataset.TARGET.vocab.itos[p] + ">__" for p in masked_pred
            ]
        g.ndata["values"][masked_nodes] = torch.tensor(
            [self.dataset.VALUES.vocab.stoi[w] for w in pred_words],
            dtype=torch.long,
            device=p_mask.device,
        )

        # update masks to mark that the positions were already predicted
        assert torch.sum(g.nodes[masked_nodes].data[mask_field]) == len(masked_nodes)
        g.ndata[mask_field][masked_nodes] = False
        if "mask_constant" in g.ndata:
            g.ndata["mask_constant"][masked_nodes] = False

        """
        Update original trees
        This is needed for adversarial attacks
        """
        assert "masked_nodes" not in g.ndata
        masked_nodes_field = torch.zeros_like(g.ndata["types"])
        g.ndata["masked_nodes"] = masked_nodes_field
        g.ndata["masked_nodes"][masked_nodes] = True
        graph_trees: List[dgl.DGLGraph] = dgl.unbatch(g)
        del g.ndata["masked_nodes"]
        batch_trees: List[AstTree] = [
            self.subtree_adversary.trees[idx] for idx in batch.ids
        ]
        for graph_tree, batch_tree in zip(graph_trees, batch_trees):
            assert graph_tree.number_of_nodes() == len(batch_tree)
            for node_id, value in zip(
                np.flatnonzero(graph_tree.ndata["masked_nodes"].cpu().numpy()),
                pred_words,
            ):
                node = batch_tree.nodes[node_id]
                # node.fields['types'] = value
                node.fields["values"] = value
                node.fields["mask_valid"] = False
                node.fields["mask_constant"] = False

        """
        Visualize Samples
        """
        if num_verbose == 0:
            return num_predicted, num_shown
        is_correct = (gts == preds).cpu()
        if num_shown >= num_verbose and is_correct.masked_select(mask).all():
            return num_predicted, num_shown

        assert "is_correct" not in g.ndata
        g.ndata["is_correct"] = is_correct
        assert "mask" not in g.ndata
        g.ndata["mask"] = mask
        trees = dgl.unbatch(g)
        del g.ndata["is_correct"]
        del g.ndata["mask"]

        samples = self.dataset.samples_for_batch(batch)
        for tree_id, tree in enumerate(trees):
            if torch.sum(tree.ndata["mask"]) == 0:
                continue
            if num_shown >= 2 * num_verbose:
                continue
            if (
                num_shown >= num_verbose
                and tree.ndata["is_correct"].masked_select(tree.ndata["mask"]).all()
            ):
                continue
            sample = samples[tree_id]
            TreeVisualization.visualize(
                sample,
                ["types", "values"],
                self.dataset.dtrain.fields,
                labels=[
                    # PredictLabel.from_iter([batch], sample, self.model, self.dataset, mask=MaskLabel.from_sample(sample, mask_field)),
                    # PredictLabel.from_values(tree.ndata['preds'], sample, self.dataset, mask=MaskLabel.from_values(tree.ndata['mask'])),
                    BooleanWithMaskLabel.from_values(
                        tree.ndata["mask"], tree.ndata["is_correct"]
                    ),
                    # TopkLabel.from_iter([batch], sample, self.model, self.dataset, mask=MaskLabel.from_values(tree.ndata['mask'])),
                    # FieldLabel(self.dataset.dtrain.fields['types'], tree.ndata['types']),
                    FieldLabel(
                        self.dataset.dtrain.fields["values"], tree.ndata["values"]
                    ),
                ],
            )
            num_shown += 1

        # Logger.debug('Number of predicted nodes: {}'.format(num_predicted))
        return num_predicted, num_shown


class RobustModelEval:
    def __init__(self, subtree_adversary, models: List[RobustModel] = None):
        self.models = models or []
        self.subtree_adversary = subtree_adversary

    def load_models(
        self,
        make_model,
        dataset: Dataset,
        adversary,
        subtree_adversary,
        args,
        model_id,
        max_models=None,
        last_base=True,
    ):
        self.models = []
        while max_models is None or len(self.models) < max_models:
            model = RobustModel(
                make_model,
                dataset,
                idx=len(self.models),
                rename_adversary=adversary,
                subtree_adversary=subtree_adversary,
                base_model=last_base and len(self.models) + 1 == max_models,
            )
            if not model.load(args, model_id):
                break
            self.models.append(model)
        Logger.debug("Loaded {} models".format(len(self.models)))
        # exit(0)

    def padding_mask(self, batch, mask_field: Optional[str] = None):
        return self.models[0].model.padding_mask(batch, mask_field=mask_field)

    def apply_batch(self, batch: MiniBatch):
        fbatch = batch.clone()
        batch_trees: List[AstTree] = [
            self.subtree_adversary.trees[idx] for idx in batch.ids
        ]

        for idx, model in enumerate(self.models):
            model.predict_with_reject(fbatch, apply=True)
        try:
            yield fbatch
        except GeneratorExit:
            return
        finally:
            "reverts modifications to the ASTs"
            for tree in batch_trees:
                for node in tree.nodes:
                    if node.fields and len(node.fields) > 1:
                        node.fields = {"target": node.fields["target"]}

    def predict_with_reject(self, batch: MiniBatch, reject_id, threshold=None):
        fbatch = batch.clone()
        batch_trees: List[AstTree] = [
            self.subtree_adversary.trees[idx] for idx in batch.ids
        ]

        preds = None
        for idx, model in enumerate(self.models):
            tmasks = model.model.padding_mask(fbatch, mask_field="mask_valid")
            batch_preds = model.predict_with_reject(
                fbatch, apply=idx != len(self.models)
            )
            if preds is None:
                preds = batch_preds
                continue

            non_rejected = (batch_preds != reject_id) & tmasks
            assert (preds[non_rejected] == reject_id).all()
            preds[non_rejected] = batch_preds[non_rejected]

        "reverts modifications to the ASTs"
        for tree in batch_trees:
            for node in tree.nodes:
                if node.fields and len(node.fields) > 1:
                    node.fields = {"target": node.fields["target"]}

        return preds

    def reset_apply(self):
        for tree in self.subtree_adversary.trees.values():
            for node in tree.nodes:
                if node.fields and len(node.fields) > 1:
                    node.fields = {"target": node.fields["target"]}


def robust_multi(
    args, dataset: Dataset, device: torch.device, masks, max_models=20, model_id=0
):
    train_iter, valid_iter, test_iter = Iterators.make(
        args, Models[args.model], dataset, device, masks
    )

    def make_model():
        return Models.make(args, dataset, device, train_iter)

    adversary, subtree_adversary = make_adversary(
        dataset,
        functools.partial(
            Iterators.make_single,
            args,
            Models[args.model],
            device,
            masks,
            dataset.EDGES,
        ),
    )

    stats = collections.Counter()
    stats["mask_valid_noreject_correct"] = 0
    stats["mask_valid_noreject_predicted"] = 0
    models = []
    for idx in range(max_models):
        # last model is trained without threshold to predict all the remaining samples
        base_model = (idx + 1) == max_models
        if not base_model:
            continue
        Logger.debug("Training iter: {}, base_model: {}".format(idx, base_model))
        if mask_count(train_iter) == 0:
            break

        model = RobustModel(
            make_model,
            dataset,
            idx=idx,
            rename_adversary=adversary,
            subtree_adversary=subtree_adversary,
            base_model=base_model,
        )

        if model_id is not None and model.load(args, model_id):
            # TODO: refactor, model is loaded but it needs to be applied on the iterator
            model.train(
                train_iter,
                valid_iter,
                num_epochs=args.num_epochs,
                test_iter=test_iter,
                apply_model=True,  # base_model=base_model,
                train_adv_mode=args.train_adv_mode,
                loaded=True,
            )
        else:
            Logger.debug(
                "Train positions to predict: {}".format(mask_count(train_iter))
            )
            Logger.debug(
                "Valid positions to predict: {}".format(mask_count(valid_iter))
            )

            model_eval = None
            if base_model:
                # reset iterators
                train_iter, valid_iter, test_iter = Iterators.make(
                    args, Models[args.model], dataset, device, masks
                )

                model_eval = RobustModelEval(subtree_adversary)
                model_eval.load_models(
                    make_model,
                    dataset,
                    adversary,
                    subtree_adversary,
                    args,
                    model_id,
                    max_models=max_models - 1,
                    last_base=False,
                )

                # train_iter = RobustModelBatchIter(model_eval, train_iter)

            if not model.train(
                train_iter,
                valid_iter,
                num_epochs=args.num_epochs,
                test_iter=test_iter,
                apply_model=True,  # base_model=base_model,
                train_adv_mode=args.train_adv_mode,
                min_nonabstained=args.min_nonabstained,
                model_eval=model_eval,
            ):
                break

            if model_id is not None:
                model.save(args, model_id)
                exit(0)

        models.append(model)

        for key in stats.keys():
            stats[key] += model.valid_stats[key]

        Logger.debug(
            "Valid Accuracy: {}/{} ({:.2f}%)".format(
                stats["mask_valid_noreject_correct"],
                stats["mask_valid_noreject_predicted"],
                acc(
                    stats["mask_valid_noreject_correct"],
                    stats["mask_valid_noreject_predicted"],
                ),
            )
        )

    return eval(args, dataset, device, masks, max_models=max_models, model_id=model_id)


def eval(
    args, dataset: Dataset, device: torch.device, masks, max_models=20, model_id=0
):
    train_iter, valid_iter, test_iter = Iterators.make(
        args, Models[args.model], dataset, device, masks
    )

    def make_model():
        return Models.make(args, dataset, device, train_iter)

    adversary, subtree_adversary = make_adversary(
        dataset,
        functools.partial(
            Iterators.make_single,
            args,
            Models[args.model],
            device,
            masks,
            dataset.EDGES,
        ),
    )

    model_eval = RobustModelEval(subtree_adversary)
    model_eval.load_models(
        make_model,
        dataset,
        adversary,
        subtree_adversary,
        args,
        model_id,
        max_models=max_models,
        last_base=args.last_base,
    )
    model_eval.reset_apply()

    test_adv_stats = eval_adversarial(
        model_eval,
        test_iter,
        adversary,
        subtree_adversary,
        n_renames=args.n_renames,
        n_subtree_renames=args.n_subtree,
        adv_mode=args.adv_mode,
        out_file=args.out_file,
        approximate=True,
    )
    d = test_adv_stats.to_dict()
    # d['valid_acc'] = valid_acc['mask_valid_acc']
    # d['test_acc'] = test_acc['mask_valid_acc']
    d = {k: [v] for k, v in d.items()}
    return pd.DataFrame(data=d, index=[model_id])


class CachedIter:
    def __init__(self, it):
        Logger.debug("Caching Batches")
        self.batches = [batch.clone() for batch in tqdm.tqdm(it)]

    def init_epoch(self):
        pass

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


class RobustModelBatchIter:
    def __init__(self, model: RobustModelEval, dataset_iter):
        self.model = model
        self.it = dataset_iter

    def init_epoch(self):
        self.it.init_epoch()

    def __len__(self):
        return len(self.it)

    def iter_batch(self, batch):
        batch_iter = (
            [batch]
            if not isinstance(self.it, AdversaryBatchIter)
            else self.it.iter_batch(batch)
        )
        for it_batch in batch_iter:
            for fbatch in self.model.apply_batch(it_batch):
                yield fbatch

    def __iter__(self):
        for batch in self.it:
            for fbatch in self.model.apply_batch(batch):
                yield fbatch


def main():
    args = parse_args()
    if not args.include_values:
        # When the values are not included renaming is a no-op
        args.n_renames = 0
    if args.adv_mode != "RANDOM" or args.train_adv_mode != "RANDOM":
        args.dot_product_embedding = True

    args.tag = "{}/robust".format(args.tag)

    """
    Debug Initialization
    """
    Logger.init(args.log_file)
    Logger.debug(" ".join(sys.argv))
    Random.seed(args.seed)

    USE_CUDA = torch.cuda.is_available() and args.use_cuda
    device = torch.device("cuda" if USE_CUDA else "cpu")

    """
    Dataset Loading and Preprocessing
    """
    dataset = Dataset(
        args,
        include_edges=args.model
        in [Models.UGraphTransformer.name, Models.GCN.name, Models.GGNN.name],
    )
    dataset.remove_duplicates()

    masks = {"mask_valid": dataset.MASK_VALID}

    """
    Training
    """
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    def save_results(data):
        data = pd.concat(data)
        print(data)
        csv_path = os.path.join(checkpoint_dir(args), "results.csv")
        data.to_csv(csv_path, index=False, header=True)

    dfs = []
    for i in range(args.repeat):
        Random.seed(args.seed + i)
        if args.eval:
            df = eval(
                args, dataset, device, masks, max_models=args.max_models, model_id=i
            )
        else:
            df = robust_multi(
                args, dataset, device, masks, max_models=args.max_models, model_id=i
            )
        dfs.append(df)
        save_results(dfs)


if __name__ == "__main__":
    main()
