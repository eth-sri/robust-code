import collections
import os
import random
from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import dgl
import numpy as np
import torch
import torchtext
import tqdm

from robustcode.analysis.graph import AstTree
from robustcode.models.modules.dgl.utransformer import GraphModel
from robustcode.models.modules.iterators import MiniBatch
from robustcode.models.modules.neural_model_base import NeuralModelBase
from robustcode.models.modules.neural_model_base import PredictionStats
from robustcode.models.robust.adversary.node_grads import each_node_grads_batch
from robustcode.models.robust.adversary.rules import NodeRenameRule
from robustcode.models.robust.adversary.rules import NodeValueIndex
from robustcode.models.robust.adversary.rules import RenameRulesForTree
from robustcode.models.robust.adversary.rules import RenameRulesIndex
from robustcode.models.robust.adversary.rules import ShuffleStrategy
from robustcode.models.robust.adversary.rules import ShuffleStrategyGradient
from robustcode.models.robust.adversary.rules import ShuffleStrategyIndividualGradient
from robustcode.models.robust.adversary.rules import ShuffleStrategyRandom
from robustcode.models.robust.adversary.tree_rules import PositionIDs
from robustcode.models.robust.adversary.tree_rules import TernaryWrapperRule
from robustcode.models.robust.dataset import Dataset
from robustcode.models.robust.dataset_util import dataset_to_trees_num
from robustcode.models.robust.visualization import FieldLabel
from robustcode.models.robust.visualization import MaskLabel
from robustcode.models.robust.visualization import TreeVisualization
from robustcode.util.misc import acc
from robustcode.util.misc import Logger
from robustcode.util.misc import Timer


class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class AdvType(OrderedEnum):
    SOUND = 1
    SOUND_PRECISE = 2
    UNSOUND = 3


class AdversarialMode(Enum):
    RANDOM = 0
    BATCH_GRADIENT_GREEDY = 11
    BATCH_GRADIENT_ASCENT = 12
    BATCH_GRADIENT_BOOSTING = 21
    INDIVIDUAL_GRADIENT = 22
    GENETIC = 31

    def __str__(self):
        return str(self.name)

    @staticmethod
    def from_string(s):
        try:
            return AdversarialMode[s]
        except KeyError:
            raise ValueError()


class AdversaryAccuracyStats:
    """
    There are several cases we want to cover:

    Original Prediction
      Correct:
        (Sound & Precise) all the adversarial samples are correct
        (Sound)           all the adversarial samples are either correct or rejected
        (Unsound)         an incorrect adversarial sample exists
       Reject:
        (Sound & Precise) all the adversarial samples are either correct or rejected
        (Sound)           all the adversarial samples are rejected
        (Unsound)         an incorrect adversarial sample exists (i.e., rejected sample classified incorrectly)
    Incorrect:
        --
    """

    def __init__(
        self, reject_token_id, out_file=None, out_field=None, unsound_token_id=None
    ):
        self.reject_token_id = reject_token_id
        self.unsound_token_id = unsound_token_id

        self.base_correct = None
        self.base_preds = None
        self.base_reject_mask = None
        self.base_y = None

        self.correct = None
        self.reject = None
        self.incorrect = None

        self.stats = collections.Counter()

        self.batch_ids = None
        self.out_file = out_file
        self.data = {}

        self.base_stats = None
        self.adv_stats = None
        if out_field is not None:
            self.base_stats = PredictionStats(out_field)
            self.adv_stats = PredictionStats(out_field)

    def training_mask(
        self,
        base_correct: torch.Tensor,
        base_preds: torch.Tensor,
        adv_correct: torch.Tensor,
        adv_preds: torch.Tensor,
    ):
        base_reject_mask = base_preds == self.reject_token_id
        adv_reject_mask = adv_preds == self.reject_token_id

        res = {
            "correct": {
                AdvType.SOUND_PRECISE: base_correct & adv_correct,
                AdvType.SOUND: base_correct & (adv_correct | adv_reject_mask),
            },
            "reject": {
                AdvType.SOUND_PRECISE: base_reject_mask
                & (adv_correct | adv_reject_mask),
                AdvType.SOUND: base_reject_mask & adv_reject_mask,
            },
            "incorrect": {AdvType.UNSOUND: ~(adv_correct | adv_reject_mask)},
        }

        res["correct"][AdvType.UNSOUND] = base_correct & ~res["correct"][AdvType.SOUND]
        res["reject"][AdvType.UNSOUND] = (
            base_reject_mask & ~res["reject"][AdvType.SOUND_PRECISE]
        )

        unsound_mask = (
            res["correct"][AdvType.UNSOUND]
            | res["reject"][AdvType.UNSOUND]
            | res["incorrect"][AdvType.UNSOUND]
        )
        imprecise_mask = (
            ~res["correct"][AdvType.SOUND_PRECISE] & res["correct"][AdvType.SOUND]
        )
        return unsound_mask, imprecise_mask

    def set_base_predictions(
        self,
        base_correct: torch.Tensor,
        base_preds: torch.Tensor,
        base_y=None,
        batch_ids=None,
    ):
        self.base_correct = base_correct
        self.base_preds = base_preds
        self.base_reject_mask = base_preds == self.reject_token_id
        self.base_y = base_y

        self.correct = {
            AdvType.SOUND_PRECISE: base_correct.clone(),
            AdvType.SOUND: base_correct.clone(),
            # AdvType.UNSOUND: torch.zeros_like(base_correct, dtype=torch.bool),
        }

        self.reject = {
            AdvType.SOUND_PRECISE: self.base_reject_mask.clone(),
            AdvType.SOUND: self.base_reject_mask.clone(),
            # AdvType.UNSOUND: torch.zeros_like(self.base_reject_mask, dtype=torch.bool),
        }

        self.incorrect = {AdvType.UNSOUND: ~(self.base_correct | self.base_reject_mask)}
        Logger.debug(
            "correct: {}, reject: {}, incorrect: {}, total: {}".format(
                self.base_correct.sum().item(),
                self.base_reject_mask.sum().item(),
                (~(self.base_correct | self.base_reject_mask)).sum().item(),
                self.base_correct.numel(),
            )
        )
        self.batch_ids = batch_ids

    def add_adversarial_predictions(self, correct: torch.Tensor, preds: torch.Tensor):
        reject_mask = preds == self.reject_token_id
        changed = False

        def update_field(field, key, value, is_and=True):
            old = field[key]
            if is_and:
                new = old & value
            else:
                new = old | value
            if torch.all(old == new):
                return False

            field[key] = new
            return True

        changed |= update_field(self.correct, AdvType.SOUND_PRECISE, correct)
        changed |= update_field(self.correct, AdvType.SOUND, (correct | reject_mask))

        changed |= update_field(
            self.reject, AdvType.SOUND_PRECISE, (correct | reject_mask)
        )
        changed |= update_field(self.reject, AdvType.SOUND, reject_mask)
        return changed

    def update_stats(self):
        for key, value in self.correct.items():
            self.stats[("correct", key)] += torch.sum(value).item()

        for key, value in self.reject.items():
            self.stats[("reject", key)] += torch.sum(value).item()

        for key, value in self.incorrect.items():
            self.stats[("incorrect", key)] += torch.sum(value).item()

        self.stats["total"] += self.base_correct.numel()
        self.stats["correct"] += torch.sum(self.base_correct).item()
        self.stats["incorrect"] += torch.sum(
            ~(self.base_correct | self.base_reject_mask)
        ).item()
        self.stats["reject"] += torch.sum(self.base_reject_mask).item()

        self.stats[("correct", AdvType.UNSOUND)] = (
            self.stats["correct"] - self.stats[("correct", AdvType.SOUND)]
        )
        self.stats[("reject", AdvType.UNSOUND)] = (
            self.stats["reject"] - self.stats[("reject", AdvType.SOUND_PRECISE)]
        )

        self.correct[AdvType.UNSOUND] = self.base_correct & ~self.correct[AdvType.SOUND]
        self.reject[AdvType.UNSOUND] = (
            self.base_reject_mask & ~self.reject[AdvType.SOUND_PRECISE]
        )
        # correct_unsound = self.base_correct & ~self.correct[AdvType.SOUND]
        # reject_unsound = self.base_reject_mask & ~self.reject[AdvType.SOUND_PRECISE]
        unsound = (
            self.incorrect[AdvType.UNSOUND]
            | self.correct[AdvType.UNSOUND]
            | self.reject[AdvType.UNSOUND]
        )
        self.stats[("total", AdvType.UNSOUND)] += torch.sum(unsound).item()
        self.stats[("total", AdvType.SOUND)] += torch.sum(
            self.correct[AdvType.SOUND] | self.reject[AdvType.SOUND_PRECISE]
        ).item()

        self.stats["total_inputs"] += self.base_correct.shape[0]  # TODO: per document
        self.stats["correct_inputs"] += torch.sum(self.base_correct, dim=0).item()

        # self.stats['reject_inputs'] += torch.sum(self.base_reject_mask)

        if self.base_stats is not None and self.base_y is not None:
            self.base_stats.add_(self.base_y, self.base_preds, self.base_correct)

            adv_preds = self.base_preds.clone()
            adv_preds[unsound] = self.unsound_token_id
            self.adv_stats.add_(self.base_y, adv_preds, ~unsound)

        if self.out_file is not None:
            self.data[str(self.batch_ids)] = {
                "correct": self.correct,
                "reject": self.reject,
                "incorrect": self.incorrect,
                "base_correct": self.base_correct,
                "base_preds": self.base_preds,
                "base_y": self.base_y,
            }

    def save(self):
        if self.out_file is None:
            return
        assert self.out_file is not None
        Logger.debug("Saving {}...".format(self.out_file))
        torch.save(self.data, "{}.pt".format(self.out_file))

    def load(self):
        assert self.out_file is not None
        assert os.path.exists("{}.pt".format(self.out_file))
        data = torch.load("{}.pt".format(self.out_file))
        self.stats = collections.Counter()
        for batch_ids, stats in data.items():
            self.base_correct = stats["base_correct"]
            self.base_preds = stats["base_preds"]
            self.base_y = stats["base_y"]
            self.base_reject_mask = self.base_preds == self.reject_token_id

            self.correct = stats["correct"]
            self.reject = stats["reject"]
            self.incorrect = stats["incorrect"]

            self.update_stats()

        return data

    def load_fallback(self, other_out_file):
        assert self.out_file is not None
        assert os.path.exists("{}.pt".format(self.out_file))
        assert os.path.exists("{}.pt".format(other_out_file))
        data = torch.load("{}.pt".format(self.out_file))
        other_data = torch.load("{}.pt".format(other_out_file))

        def merge(A, B, reject):
            return A * ~reject + B * reject

        self.stats = collections.Counter()
        for batch_ids, stats in data.items():
            self.base_correct = stats["base_correct"]
            self.base_preds = stats["base_preds"]
            self.base_y = stats["base_y"]
            self.base_reject_mask = self.base_preds == self.reject_token_id

            self.correct = stats["correct"]
            self.reject = stats["reject"]
            self.incorrect = stats["incorrect"]

            ostats = other_data[batch_ids]
            obase_correct = ostats["base_correct"]
            obase_preds = ostats["base_preds"]

            ocorrect = ostats["correct"]
            oreject = ostats["reject"]
            oincorrect = ostats["incorrect"]

            self.base_correct = merge(
                self.base_correct, obase_correct, self.base_reject_mask
            )
            self.base_preds = merge(self.base_preds, obase_preds, self.base_reject_mask)

            for key in self.correct.keys():
                self.correct[key] = merge(
                    self.correct[key], ocorrect[key], self.base_reject_mask
                )

            for key in self.reject.keys():
                self.reject[key] = merge(
                    self.reject[key], oreject[key], self.base_reject_mask
                )

            for key in self.incorrect.keys():
                self.incorrect[key] = merge(
                    self.incorrect[key], oincorrect[key], self.base_reject_mask
                )

            self.base_reject_mask[:] = False

            self.update_stats()

    def is_sound(self):
        return (
            self.stats[("correct", AdvType.UNSOUND)]
            + self.stats[("reject", AdvType.UNSOUND)]
            + self.stats[("incorrect", AdvType.UNSOUND)]
            == 0
        )

    def to_dict(self):
        stats = {}
        shortcut = {"correct": "C", "incorrect": "I", "reject": "R", "total": "T"}
        for key in sorted(key for key in self.stats.keys() if not isinstance(key, str)):
            category, name = key
            stats["{}:{}".format(shortcut[category], name.name)] = acc(
                self.stats[key], self.stats[category]
            )
            if category not in stats:
                stats["{} ({})".format(category, shortcut[category])] = self.stats[
                    category
                ]
        return stats

    def __repr__(self):
        s = "Base Accuracy: {:6d}/{:6d} ({:.2f}%), reject: {:6d}/{:6d} ({:.2f}%)\n".format(
            self.stats["correct"],
            self.stats["total"] - self.stats["reject"],
            acc(self.stats["correct"], self.stats["total"] - self.stats["reject"]),
            self.stats["reject"],
            self.stats["total"],
            acc(self.stats["reject"], self.stats["total"]),
        )
        for key in sorted(key for key in self.stats.keys() if not isinstance(key, str)):
            category, name = key
            s += "\t{:>10s} {:>20s}: {:6d}/{:6d} ({:.2f}%)\n".format(
                category,
                name.name,
                self.stats[key],
                self.stats[category],
                acc(self.stats[key], self.stats[category]),
            )

        #     Logger.debug('Most common predictions:')
        if self.base_stats is not None:
            self.base_stats.dump_most_common("correct", 10)
            self.adv_stats.dump_most_common("unsound", 10)
        return (
            s
            + f"\nCorrect inputs: {self.stats['correct_inputs']} / {self.stats['total_inputs']}"
        )


class AdversaryBase(ABC):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def attack(
        self,
        model: NeuralModelBase,
        batch: MiniBatch,
        mask=None,
        num_samples=10,
        adv_mode=AdversarialMode.RANDOM,
        shuffle: Optional[ShuffleStrategy] = None,
    ):
        raise NotImplementedError

    def compute_correct(
        self, model: NeuralModelBase, batch: MiniBatch, verbose=False, threshold=0.5
    ):
        training_mask = model.padding_mask(batch)
        mask_valid = batch.masks["mask_valid"]
        Y = batch.Y
        predictions = model.predict_with_reject(
            batch, self.dataset.reject_token_id, threshold=threshold
        )

        correct = (predictions == Y) * training_mask
        if len(Y.size()) == 2:
            """
            The adversary assumes that the order of predicted labels in the adversary and original batch are the same.
            All the modifications respect this, i.e., while code is added or removed, the order and 
            number of predicted labels (via 'mask_valid') stays the same.
            However, since the sample length might change, we need to ensure that masked_select is performed
            batch_first (B x N rather than N x B) as otherwise the results are no guaranteed to preserve the order.              
            """
            # TODO: check that the batch is created with batch_first=False (this is the default with torchtext)
            mask_valid = mask_valid.t()
            correct = correct.t().masked_select(mask_valid)
            predictions = predictions.t().masked_select(mask_valid)
            Y = Y.t().masked_select(mask_valid)
        else:
            correct = correct.masked_select(mask_valid)
            predictions = predictions.masked_select(mask_valid)
            Y = Y.masked_select(mask_valid)
        return correct, predictions, Y

    def visualize_adversarial(
        self, batch: MiniBatch, model: NeuralModelBase, masks, colors
    ):
        assert isinstance(model, GraphModel)
        samples = self.dataset.samples_for_batch(batch)

        training_mask = model.padding_mask(batch, mask_field="mask_valid")

        g = batch.X
        mask_data = {}
        mask_names = []
        for idx, mask in enumerate(masks):
            mask_all = torch.zeros_like(training_mask)
            mask_all[training_mask] = mask
            mask_data[str(idx)] = mask_all
            mask_names.append(str(idx))

        for name, mask in mask_data.items():
            assert name not in g.ndata
            g.ndata[name] = mask
        trees = dgl.unbatch(g)
        for name in mask_data.keys():
            del g.ndata[name]

        for tree, sample in zip(trees, samples):
            mask_labels = [
                MaskLabel.from_values(tree.ndata[name], color)
                for name, color in zip(mask_names, colors)
            ]
            TreeVisualization.visualize(
                sample,
                ["types", "values"],
                self.dataset.dtrain.fields,
                labels=mask_labels
                + [
                    FieldLabel(
                        self.dataset.dtrain.fields["types"], tree.ndata["types"]
                    ),
                    FieldLabel(
                        self.dataset.dtrain.fields["values"], tree.ndata["values"]
                    ),
                    # TODO: topk
                    # TopkLabel.from_iter([batch], sample, model, self.dataset,
                    #                     mask=MaskLabel.from_sample(sample, 'mask_valid')),
                ],
            )
            input()
            break

    def compute_gradients(
        self, batch, model, unsound_masks, imprecise_masks, threshold
    ):
        masks = [a | b for a, b in zip(unsound_masks, imprecise_masks)]

        for node_grads in each_node_grads_batch(
            batch,
            model,
            self.dataset,
            model.loss_function,
            threshold=threshold,
            masks=masks,
            max_samples=None,
        ):
            print(node_grads)

    def fit_adversarial(
        self,
        model: NeuralModelBase,
        dataset_iter: Iterable[MiniBatch],
        adversarial_iters: Union["AdversaryBatchIter", List["AdversaryBatchIter"]],
        threshold=0.5,
    ):
        if not isinstance(adversarial_iters, list):
            adversarial_iters = [adversarial_iters]
        num_refined = 0
        num_unsound = 0
        num_imprecise = 0

        def sanity_check(orig_batch: MiniBatch, adversarial_batch: MiniBatch):
            gts = orig_batch.Y
            tmasks = model.padding_mask(orig_batch, mask_field="mask_valid")

            adv_gts = adversarial_batch.Y
            adv_tmasks = model.padding_mask(adversarial_batch, mask_field="mask_valid")
            assert torch.sum(tmasks) == torch.sum(adv_tmasks), "{} vs {}".format(
                torch.sum(tmasks), torch.sum(adv_tmasks)
            )
            assert torch.all(gts[tmasks] == adv_gts[adv_tmasks])

        adv_stats = AdversaryAccuracyStats(self.dataset.reject_token_id)
        for batch in tqdm.tqdm(dataset_iter, ncols=100, leave=False):
            base_correct, base_preds, base_y = self.compute_correct(
                model, batch, verbose=False, threshold=threshold
            )

            # self.visualize_adversarial(batch, model, [base_correct], ['da_green'])
            for adversarial_iter in adversarial_iters:
                for idx, adv_batch in enumerate(adversarial_iter.iter_batch(batch)):
                    sanity_check(batch, adv_batch)

                    correct, preds, adv_y = self.compute_correct(
                        model, adv_batch, verbose=False, threshold=threshold
                    )
                    assert torch.all(base_y == adv_y)
                    unsound_mask, imprecise_mask = adv_stats.training_mask(
                        base_correct, base_preds, correct, preds
                    )
                    # self.compute_gradients(adv_batch, model, unsound_mask, imprecise_mask, threshold)

                    mask = unsound_mask | imprecise_mask
                    num_refined += torch.sum(mask).item()
                    num_unsound += torch.sum(unsound_mask).item()
                    num_imprecise += torch.sum(imprecise_mask).item()

                    # self.visualize_adversarial(adv_batch, model, [unsound_mask, imprecise_mask], ['da_red', 'da_black'])

                    model.fit_batch(
                        model.opt,
                        model.loss_function,
                        adv_batch,
                        mask_field="mask_valid",
                        training_mask=mask,
                    )

        Logger.debug(
            "Number of refined samples: {}, unsound: {}, imprecise: {}".format(
                num_refined, num_unsound, num_imprecise
            )
        )
        return num_refined

    def adversarial_accuracy(
        self,
        model: NeuralModelBase,
        base_iter: Iterable[MiniBatch],
        adversarial_iters: Union["AdversaryBatchIter", List["AdversaryBatchIter"]],
        threshold=0.5,
        out_file=None,
        verbose=False,
        approximate=False,
    ) -> AdversaryAccuracyStats:
        if not isinstance(adversarial_iters, list):
            adversarial_iters = [adversarial_iters]

        Logger.debug("out_file: {}".format(out_file))
        adv_stats = AdversaryAccuracyStats(
            self.dataset.reject_token_id,
            out_file=out_file,
            out_field=self.dataset.TARGET,
            unsound_token_id=self.dataset.unsound_token_id,
        )
        for batch_idx, batch in enumerate(base_iter):
            # if approximate and random.random() > 0.3:
            #     continue
            if approximate and batch_idx % 4 != 0:
                continue
            base_correct, base_preds, base_y = self.compute_correct(
                model, batch, verbose=False, threshold=threshold
            )
            adv_stats.set_base_predictions(
                base_correct, base_preds, base_y=base_y, batch_ids=batch.ids
            )

            "Combine results of multiple possible adversarial attacks to obtain the worst case results"
            for adversarial_iter in adversarial_iters:
                changed = False
                for idx, adv_batch in enumerate(adversarial_iter.iter_batch(batch)):
                    correct, preds, adv_y = self.compute_correct(
                        model, adv_batch, verbose=False, threshold=threshold
                    )
                    assert torch.all(
                        base_y == adv_y
                    ), "The order of predicted positions changed. The adversary will compute incorrect results!"
                    changed |= adv_stats.add_adversarial_predictions(correct, preds)

                    if idx > 0 and idx % 20 == 0:
                        if not changed:
                            Logger.debug("Stopping after {} iter".format(idx))
                            break
                        changed = False
                        Logger.debug(
                            "\titer {} {}".format(
                                idx,
                                str(
                                    [v.sum().item() for v in adv_stats.correct.values()]
                                ),
                            )
                        )

            adv_stats.update_stats()

            if verbose:
                Logger.debug(
                    "Batch: {}, Adv Accuracy\n{}".format(batch_idx, str(adv_stats))
                )
        adv_stats.save()

        Logger.debug("Adv Accuracy\n{}".format(str(adv_stats)))
        return adv_stats


class RenameAdversary(AdversaryBase):
    def __init__(self, rules_index: RenameRulesIndex, dataset: Dataset):
        super().__init__(dataset)
        self.index = rules_index

    def attack(
        self,
        model: NeuralModelBase,
        batch: MiniBatch,
        mask=None,
        num_samples=10,
        adv_mode=AdversarialMode.RANDOM,
        shuffle: Optional[ShuffleStrategy] = None,
    ):
        if num_samples == 0:
            return
        shuffle = shuffle or ShuffleStrategyRandom()
        if isinstance(batch.X, dgl.DGLGraph):
            yield from self.__attack_graph(
                model, batch, mask, num_samples, shuffle, mode=adv_mode
            )
        else:
            yield from self.__attack_seq(
                model, batch, mask, num_samples, shuffle, mode=adv_mode
            )

    def __attack_seq(
        self,
        model: NeuralModelBase,
        batch: MiniBatch,
        mask: torch.Tensor,
        num_samples,
        shuffle: ShuffleStrategy,
        mode: AdversarialMode,
    ):

        tree_sizes = batch.lengths
        if mask is not None:
            raise NotImplementedError
        else:
            tree_nodes = [None] * len(tree_sizes)

        tree_rules, tree_num_samples = self._initialize_adversarial_rules(
            num_samples, tree_nodes, model, batch, shuffle=shuffle, mode=mode
        )

        # the inputs should contain two tensors for types and values
        # assert len(batch.inputs) == 2
        # only values are being changed
        original_values = batch.X[-1]
        adversarial_masks = None

        try:
            for idx in range(max(tree_num_samples)):
                batch_values = batch.X[-1].cpu().t().numpy()
                idx_to_use = idx

                if mode == AdversarialMode.INDIVIDUAL_GRADIENT:
                    # obtain gradients w.r.t. idx-th classifiable position in each input
                    shuffle.for_next_position()
                    for rules in tree_rules:
                        for rule in rules:
                            shuffle.shuffle_candidates(rule)

                if mode == AdversarialMode.BATCH_GRADIENT_BOOSTING:
                    # mask: compute gradient only for correctly predicted positions in all previous iterations
                    batch.X[-1] = original_values
                    adversarial_masks = [
                        model.get_adversarial_mask(
                            minibatch,
                            mask_field="mask_valid",
                            previous_mask=adversarial_masks,
                        )
                        for minibatch in batch
                    ]

                if mode in [
                    AdversarialMode.BATCH_GRADIENT_ASCENT,
                    AdversarialMode.BATCH_GRADIENT_BOOSTING,
                ]:
                    shuffle.initialize(
                        model, batch, None, position_mask=adversarial_masks
                    )
                    for rules in tree_rules:
                        for rule in rules:
                            shuffle.shuffle_candidates(rule)
                    # values have been shuffled again, we can just use the argmax
                    idx_to_use = 0

                assert len(batch_values) == len(tree_rules)
                # for every example in batch ...
                for rules, values in zip(tree_rules, batch_values):
                    # one node with constant assign => one rule
                    for rule in rules:
                        rule.apply_first_valid(idx_to_use, values)

                batch.X[-1] = torch.tensor(
                    batch_values.transpose(),
                    dtype=torch.long,
                    device=original_values.device,
                )
                yield batch
        except GeneratorExit:
            return
        finally:
            batch.X[-1] = original_values

            for rules in tree_rules:
                for rule in rules:
                    rule.reset()

    def __attack_graph(
        self,
        model: NeuralModelBase,
        batch: MiniBatch,
        mask: torch.Tensor,
        num_samples,
        shuffle: ShuffleStrategy,
        mode: AdversarialMode,
    ):
        tree_sizes = batch.lengths
        offsets = (np.cumsum(tree_sizes) - np.array(tree_sizes)).tolist()

        if mask is not None:
            tree_masks = torch.split(mask, tree_sizes)
            tree_nodes = [
                np.flatnonzero(tree_mask.cpu().numpy()) for tree_mask in tree_masks
            ]
        else:
            tree_nodes = [None] * len(tree_sizes)

        g = batch.X
        assert g.number_of_nodes() == sum(tree_sizes)
        original_values = g.ndata["values"]

        tree_rules, tree_num_samples = self._initialize_adversarial_rules(
            num_samples, tree_nodes, model, batch, shuffle, mode
        )
        adversarial_mask = None

        try:
            for idx in range(max(tree_num_samples)):
                values = g.ndata["values"].cpu().numpy()

                if mode == AdversarialMode.INDIVIDUAL_GRADIENT:
                    # obtain gradients w.r.t. idx-th classifiable position in each input
                    shuffle.for_next_position()
                    for rules in tree_rules:
                        for rule in rules:
                            shuffle.shuffle_candidates(rule)

                if mode == AdversarialMode.BATCH_GRADIENT_BOOSTING:
                    # mask: compute gradient only for correctly predicted positions in all previous iterations
                    g.ndata["values"] = original_values
                    adversarial_mask = model.get_adversarial_mask(
                        batch, mask_field="mask_valid", previous_mask=adversarial_mask
                    )

                if mode in [
                    AdversarialMode.BATCH_GRADIENT_ASCENT,
                    AdversarialMode.BATCH_GRADIENT_BOOSTING,
                ]:
                    shuffle.initialize(model, batch, position_mask=adversarial_mask)
                    for rules in tree_rules:
                        for rule in rules:
                            shuffle.shuffle_candidates(rule)

                for ith_tree, (rules, offset) in enumerate(zip(tree_rules, offsets)):
                    # one node with constant assign => one rule
                    for rule in rules:
                        # apply each rule with the batch offset of its example
                        idx_to_use = idx
                        if mode in [
                            AdversarialMode.BATCH_GRADIENT_ASCENT,
                            AdversarialMode.BATCH_GRADIENT_BOOSTING,
                        ]:
                            # values have been shuffled again, we can just use the argmax
                            idx_to_use = 0
                        rule.apply_first_valid(idx_to_use, values, usage_offset=offset)

                g.ndata["values"] = torch.tensor(
                    values, dtype=torch.long, device=original_values.device
                )
                yield batch
        except GeneratorExit:
            return
        finally:
            g.ndata["values"] = original_values

    def _initialize_adversarial_rules(
        self,
        num_samples,
        tree_nodes,
        model,
        batch,
        shuffle: ShuffleStrategy,
        mode: AdversarialMode,
    ):
        ids = batch.ids
        rule_sets = [self.index.per_tree_rules.get(idx, None) for idx in ids]
        assert all(rule_set is not None for rule_set in rule_sets)

        if mode in [AdversarialMode.INDIVIDUAL_GRADIENT, AdversarialMode.RANDOM]:
            # initialize / compute gradients once
            shuffle.initialize(model, batch)

        do_shuffle = shuffle if (mode == AdversarialMode.RANDOM) else None

        tree_rules, tree_num_samples = zip(
            *[
                rule_set.init_rules(nodes, num_samples, shuffle=do_shuffle)
                for rule_set, nodes in zip(rule_sets, tree_nodes)
            ]
        )

        return tree_rules, tree_num_samples


class RenameAdversaryLimited(RenameAdversary):

    """
    A version of RenameAdversary that only renames values of new nodes that were created by other attacks.
    """

    def __init__(self, dataset: Dataset, trees: Dict[int, AstTree]):
        rules_index = RenameRulesIndex()
        self.trees = trees
        # obtain the list of all observed values in training dataset
        trees_train_num = dataset_to_trees_num(dataset.dtrain)
        self.value_index = NodeValueIndex(dataset, trees_train_num)
        super().__init__(rules_index, dataset)

    def attack(
        self,
        model,
        batch,
        mask=None,
        num_samples=10,
        adv_mode=AdversarialMode.RANDOM,
        shuffle: Optional[ShuffleStrategy] = None,
    ):
        ids = batch.ids
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()

        # the nodes provided by SubtreeAdversary will be different for each adv. example,
        # so we need to create new rules for every batch
        self.index = RenameRulesIndex()
        rule_sets = []
        for tree_id in ids:
            tree = self.trees[tree_id]
            rules = RenameRulesForTree(tree_id, tree)
            for pos, node in enumerate(tree.nodes):
                # create a rename rule for each valid newly generated node
                if node.origin == PositionIDs.ADVERSARIAL_CONSTANT:
                    rule = NodeRenameRule(
                        tree_id,
                        node.id,
                        [node.id],
                        [],
                        value="{:3d} {}".format(
                            self.dataset.VALUES.vocab.stoi[node.value], node.value
                        ),
                        candidate_values=self.value_index.values_for_type(node.type),
                        fixed_value_offset=self.dataset.fixed_value_offset,
                    )

                    rules.add(rule)
            self.index.add(rules)
        assert all(rule_set is not None for rule_set in rule_sets)

        return super().attack(model, batch, mask, num_samples, adv_mode, shuffle)


class SubtreeAdversary(AdversaryBase):
    def __init__(
        self,
        rules: List[TernaryWrapperRule],
        dataset: Dataset,
        trees: Dict[int, AstTree],
        make_iter,
    ):
        super().__init__(dataset)
        self.rules = rules
        self.trees = trees
        for key, value in dataset.ID.vocab.stoi.items():
            if key.endswith("_mod"):
                assert value not in self.trees
                idx = dataset.ID.vocab.stoi.get(key[:-4], None)
                assert idx is not None
                "if the tree was a duplicate it was removed from the dataset"
                if idx in self.trees:
                    self.trees[value] = self.trees[idx]
        self.make_iter = make_iter

        self.dataset = dataset
        self.timers = collections.defaultdict(Timer)

    def attack(
        self,
        model,
        batch,
        mask=None,
        num_samples=10,
        adv_mode=AdversarialMode.RANDOM,
        shuffle: Optional[ShuffleStrategy] = None,
    ):
        if adv_mode != AdversarialMode.RANDOM:
            raise NotImplementedError(
                "For subtree attack, only random adversarial mode is supported."
            )
            # TODO: different adv. modes for the two attacks

        if num_samples == 0:
            return

        tree_ids = batch.ids
        batch_trees = [self.trees[idx] for idx in tree_ids]

        per_node_rules = []
        for tree_id, tree in zip(tree_ids, batch_trees):
            for node in tree.nodes:
                candidate_rules = [rule for rule in self.rules if rule.matches(node)]
                if candidate_rules:
                    per_node_rules.append((tree_id, node, candidate_rules))

        vbatch_ids = [sample.id for sample in self.dataset.samples_for_batch(batch)]
        threshold = random.uniform(0.1, 0.4)
        for idx in range(num_samples):
            samples = []
            try:
                num_applied = 0
                for (tree_id, node, candidate_rules) in per_node_rules:
                    "apply the rule with prob threshold"
                    if random.random() <= threshold:
                        rule = random.choice(candidate_rules)
                        rule.apply(tree_id, node)
                        num_applied += 1
                # Logger.debug('Number of applied rules {}'.format(num_applied))

                for tree_id, tree in zip(tree_ids, batch_trees):
                    samples.append(self.dataset.tree_to_sample(tree_id, tree))

                dbatch = torchtext.data.Dataset(samples, self.dataset.fields)

                diter = self.make_iter(dbatch, cached=False)
                for adv_batch in diter:
                    vadv_batch_ids = [
                        sample.id[:-4]
                        for sample in self.dataset.samples_for_batch(adv_batch)
                    ]
                    assert vbatch_ids == vadv_batch_ids

                    yield adv_batch
            except GeneratorExit:
                return
            finally:
                for rule in self.rules:
                    rule.revert_all_changes()

                for tree in batch_trees:
                    tree.refresh()  # revert changes


class AdversaryBatchIter:
    def __init__(
        self,
        adversary: AdversaryBase,
        model: NeuralModelBase,
        dataset_iter,
        shuffle: ShuffleStrategy = None,
        num_samples=1,
        adv_mode=AdversarialMode.RANDOM,
    ):
        self.adversary = adversary
        self.model = model
        self.it = dataset_iter
        if adv_mode == AdversarialMode.RANDOM:
            self.shuffle = ShuffleStrategyRandom()
        elif adv_mode in [
            AdversarialMode.BATCH_GRADIENT_GREEDY,
            AdversarialMode.BATCH_GRADIENT_ASCENT,
            AdversarialMode.BATCH_GRADIENT_BOOSTING,
        ]:
            self.shuffle = ShuffleStrategyGradient()
        elif adv_mode == AdversarialMode.INDIVIDUAL_GRADIENT:
            self.shuffle = ShuffleStrategyIndividualGradient(num_samples)
        else:
            raise NotImplementedError(
                "Adversarial Mode " + str(adv_mode) + " is not supported"
            )

        self.num_samples = num_samples
        self.adversarial_mode = adv_mode

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
            for adv_batch in self.adversary.attack(
                self.model,
                it_batch,
                num_samples=self.num_samples,
                adv_mode=self.adversarial_mode,
                shuffle=self.shuffle,
            ):
                yield adv_batch

    def __iter__(self):
        for batch in self.it:
            for idx, adv_batch in enumerate(
                self.adversary.attack(
                    self.model,
                    batch,
                    num_samples=self.num_samples,
                    adv_mode=self.adversarial_mode,
                    shuffle=self.shuffle,
                )
            ):
                yield adv_batch
