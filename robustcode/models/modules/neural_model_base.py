import collections
import itertools
import os
from enum import Enum
from types import SimpleNamespace
from typing import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torchtext
import tqdm
from torch import Tensor

from robustcode.models.modules.iterators import MiniBatch
from robustcode.models.modules.util import AdaptiveLogSoftmax
from robustcode.models.modules.util import Generator
from robustcode.util.misc import acc
from robustcode.util.misc import Logger


class PredictionStats:
    def __init__(self, out_field: torchtext.data.field.Field):
        self.vocab = out_field.vocab
        self.d_vocab = len(self.vocab)
        self.correct_counts = torch.zeros(self.d_vocab)
        self.total_counts = torch.zeros(self.d_vocab)
        self.predicted_counts = torch.zeros(self.d_vocab)

    def add_(self, Y, b_prediction, is_correct):
        self.total_counts.add_(
            torch.histc(Y.cpu().float(), bins=self.d_vocab, min=0, max=self.d_vocab - 1)
        )
        self.correct_counts.add_(
            torch.histc(
                (b_prediction.masked_select(is_correct)).cpu().float(),
                bins=self.d_vocab,
                min=0,
                max=self.d_vocab - 1,
            )
        )
        self.predicted_counts.add_(
            torch.histc(
                b_prediction.cpu().float(),
                bins=self.d_vocab,
                min=0,
                max=self.d_vocab - 1,
            )
        )

    def dump_most_common(self, name, N):
        prediction_stats = [
            (predicted, correct, total, idx)
            for idx, predicted, correct, total in zip(
                itertools.count(),
                self.predicted_counts.numpy(),
                self.correct_counts.numpy(),
                self.total_counts.numpy(),
            )
        ]
        # remove unk and padding
        prediction_stats = prediction_stats[2:] + [prediction_stats[0]]
        prediction_stats.sort(reverse=True)

        Logger.debug(name)
        for predicted, correct, total, idx in prediction_stats[:N]:
            Logger.debug(
                "\t{:>30s} precision: {:10d}/{:10d} {:6.2f}%, recall: {:10d}/{:10d} {:6.2f}%".format(
                    repr(self.vocab.itos[idx]),
                    int(correct),
                    int(predicted),
                    acc(correct, predicted),
                    int(correct),
                    int(total),
                    acc(correct, total),
                )
            )

        correct = int(sum(correct for _, correct, _, _ in prediction_stats))
        total = int(sum(total for _, _, total, _ in prediction_stats))
        Logger.debug(
            "\t{:>30s}  accuracy: {:10d}/{:10d} {:6.2f}%".format(
                "total", correct, total, acc(correct, total)
            )
        )

    def get_precision(self, include_reject=True):
        prediction_stats = [
            (predicted, correct, total, idx)
            for idx, predicted, correct, total in zip(
                itertools.count(),
                self.predicted_counts.numpy(),
                self.correct_counts.numpy(),
                self.total_counts.numpy(),
            )
        ]
        # remove unk and padding
        prediction_stats = prediction_stats[2:] + [prediction_stats[0]]
        if not include_reject:
            prediction_stats = [
                (predicted, correct, total, idx)
                for predicted, correct, total, idx in prediction_stats
                if self.vocab.itos[idx] != "reject"
            ]

        correct = int(sum(correct for _, correct, _, _ in prediction_stats))
        predicted = int(sum(predicted for predicted, _, _, _ in prediction_stats))
        return SimpleNamespace(
            correct=correct, predicted=predicted, precision=(acc(correct, predicted))
        )


class AccuracyStats:
    def __init__(self, out_field: torchtext.data.field.Field):
        assert out_field.pad_token is not None
        assert out_field.unk_token is not None
        self.num_correct = 0
        self.num_samples = 0
        self.num_unk_preds = 0
        self.pad_token_id = out_field.vocab.stoi[out_field.pad_token]
        self.unk_token_id = out_field.vocab.stoi[out_field.unk_token]

    def add_(
        self, Y, b_prediction, is_correct, prob=None, threshold=None, reject_id=None
    ):
        if reject_id is not None:
            mask = b_prediction != reject_id
            Y = torch.masked_select(Y, mask)
            b_prediction = torch.masked_select(b_prediction, mask)
            is_correct = torch.masked_select(is_correct, mask)

            if prob is not None:
                prob = torch.masked_select(prob, mask)

        if prob is not None and threshold is not None:
            mask = prob >= threshold
            Y = torch.masked_select(Y, mask)
            b_prediction = torch.masked_select(b_prediction, mask)
            is_correct = torch.masked_select(is_correct, mask)
        self.num_correct += torch.sum(is_correct).item()
        self.num_unk_preds += torch.sum(
            torch.eq(b_prediction, self.unk_token_id)
        ).item()
        self.num_samples += torch.sum(torch.ne(Y, self.pad_token_id)).item()

    def get_accuracy(self):
        return acc(self.num_correct, self.num_samples)

    def __str__(self):
        return "{:8d}/{:8d} ({:5.2f}%)".format(
            self.num_correct, self.num_samples, acc(self.num_correct, self.num_samples)
        )


class FitStats:
    def __init__(self, pbar):
        self.total_loss = 0
        self.act_loss = 0
        self.num_correct = 0
        self.processed_samples = 0
        self.pbar = pbar


class DisableDropout:
    """
    Temporarily disables dropout while allowing the model to be in the training mode.
    Used for computing gradients.

    Usage:

    with DisableDropout(model):
       ...

    """

    def __init__(self, model):
        self.model = model
        self.training = model.training
        self.dropouts = []

        if not self.training:
            # nothing to do, already in eval mode
            return

        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
            elif isinstance(module, nn.RNNBase):
                self.dropouts.append(module.dropout)
                module.dropout = 0

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        if not self.training:
            return

        idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
            elif isinstance(module, nn.RNNBase):
                module.dropout = self.dropouts[idx]
                idx += 1


class NeuralModelBase(nn.Module):
    """
    Base model for Neural Network that provides functions for training and prediction
    The subclass defines the network architecture except the last softmax layer
    """

    class SoftmaxType(Enum):
        linear = 0
        adaptive = 1

    def __init__(
        self,
        d_model,
        d_out_vocab,
        softmax_type="linear",
        unk_token_id=0,
        pad_token_id=1,
    ):
        super(NeuralModelBase, self).__init__()
        self.d_model = d_model
        self.d_out_vocab = d_out_vocab

        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        self.softmax_type = NeuralModelBase.SoftmaxType[softmax_type]
        if self.softmax_type == NeuralModelBase.SoftmaxType.adaptive:
            self.softmax = AdaptiveLogSoftmax(
                d_model,
                d_out_vocab,
                cutoffs=[round(d_out_vocab / 15), 3 * round(d_out_vocab / 15)],
                div_value=4,
            )
        elif self.softmax_type == NeuralModelBase.SoftmaxType.linear:
            assert self.softmax_type == NeuralModelBase.SoftmaxType.linear
            self.softmax = Generator(d_model, d_out_vocab)

        self.attack = None
        self.loss_function = None  # loss function used for training
        self.opt = None  # optimizer used for training

    def load(self, filename):
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename))
            return True
        return False

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        Logger.debug("# Parameters: {}".format(total_params))
        Logger.debug("# Trainable Parameters: {}".format(total_trainable_params))

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def softmax_loss(
        self, h: Tensor, batch: MiniBatch, loss_function: Optional[nn.Module] = None
    ) -> Tensor:
        h = h.view(-1, h.size(-1))
        Y = batch.Y.view(-1)
        if self.softmax_type == NeuralModelBase.SoftmaxType.adaptive:
            output, _ = self.softmax(h, Y)
            loss: Tensor = -output
        else:
            assert self.softmax_type == NeuralModelBase.SoftmaxType.linear
            logits = self.softmax(h)
            loss: Tensor = loss_function(logits, Y)
        return loss.view_as(batch.Y)

    def padding_mask(
        self, batch: MiniBatch, mask_field: Optional[str] = None
    ) -> Tensor:
        # Assign zero loss to padded predictions
        mask = batch.Y != self.pad_token_id
        if mask_field in batch.masks:
            mask = mask * batch.masks[mask_field]
        return mask

    def fit_batch(
        self,
        opt: torch.optim.Optimizer,
        loss_function: nn.Module,
        batch: MiniBatch,
        mask_field: Optional[str] = None,
        stats=None,
        training_mask: Optional[Tensor] = None,
    ):
        mask = self.padding_mask(batch, mask_field)
        num_samples = mask.sum().item()
        if num_samples == 0:
            return

        out = self(batch.X)
        loss = self.softmax_loss(out, batch, loss_function=loss_function)
        loss = loss.masked_select(mask)
        if training_mask is not None:
            training_mask = training_mask.view(-1)
            assert loss.numel() == training_mask.numel()
            loss = loss.masked_select(training_mask)
        loss = loss.mean()

        if hasattr(self, "act_loss") and self.act_loss is not None:
            act_loss = self.act_loss.masked_select(mask)
            act_loss = act_loss.mean()
            loss += act_loss

            if stats is not None:
                stats.act_loss += act_loss.item() * num_samples

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        opt.step()

        if stats is not None:
            stats.total_loss += loss.item() * num_samples
            stats.processed_samples += num_samples

    def fit(self, dataset_iter, opt, loss_function, mask_field=None):
        self.train()
        if hasattr(self, "reset_stat"):
            self.reset_stat()
        dataset_iter.init_epoch()
        with tqdm.tqdm(total=len(dataset_iter), ncols=140, leave=False) as pbar:
            stats = FitStats(pbar)
            for i, batch in enumerate(dataset_iter):
                pbar.update(1)
                self.fit_batch(
                    opt, loss_function, batch, mask_field=mask_field, stats=stats
                )
                pbar.set_postfix(
                    loss="{:.4f}".format(stats.total_loss / stats.processed_samples)
                )

        Logger.debug("loss={:.4f}".format(stats.total_loss / stats.processed_samples))
        if hasattr(self, "print_stat"):
            self.print_stat()
        return stats.total_loss / stats.processed_samples

    """
    Predictions
    """

    def predict(self, batch: MiniBatch) -> Tensor:
        self.eval()
        with torch.no_grad():
            h = self(batch.X)
            return self.softmax.predict(h)

    def predict_probs_with_reject(self, batch: MiniBatch, reject_id=None):
        self.eval()
        with torch.no_grad():
            assert (
                self.softmax_type == NeuralModelBase.SoftmaxType.linear
            ), "Only Linear softmax is currently supported!"
            h = self(batch.X)
            log_probs = self.softmax.log_prob(h)

            reject_probs = None
            if reject_id is not None:
                reject_probs = torch.exp(log_probs[..., reject_id])
                # set the reject probability to zero such that the top prediction is from the vocab
                log_probs[..., reject_id] = float("-inf")
            best_probs, best_predictions = log_probs.topk(k=1, dim=-1)
            best_probs = best_probs.squeeze(dim=-1)
            best_predictions = best_predictions.squeeze(dim=-1)

            torch.exp(best_probs, out=best_probs)

            return best_probs, best_predictions, reject_probs

    def predict_with_reject(self, batch: MiniBatch, reject_id, threshold=0.5) -> Tensor:
        best_probs, best_predictions, reject_probs = self.predict_probs_with_reject(
            batch, reject_id=reject_id
        )
        best_predictions[reject_probs >= threshold] = reject_id
        return best_predictions

    def accuracy_with_reject(
        self,
        dataset_iter: Iterable[MiniBatch],
        out_field: torchtext.data.field.Field,
        reject_id=None,
        threshold=0.5,
        verbose=True,
    ):
        self.eval()
        with torch.no_grad():
            pred_stats = collections.defaultdict(lambda: PredictionStats(out_field))
            acc_stats = collections.defaultdict(lambda: AccuracyStats(out_field))

            for batch in tqdm.tqdm(dataset_iter, ncols=140, leave=False):
                if reject_id is not None:
                    b_prediction = self.predict_with_reject(
                        batch, reject_id, threshold=threshold
                    )
                else:
                    b_prediction = self.predict(batch)
                Y = batch.Y

                # Do not count padding as correct predictions
                mask = self.padding_mask(batch)
                is_correct = (b_prediction == Y) * mask

                pred_stats["all"].add_(Y, b_prediction, is_correct)
                acc_stats["all"].add_(Y, b_prediction, is_correct)

                for mask_name, bmask in batch.masks.items():
                    mY = torch.masked_select(Y, bmask)
                    mb_prediction = torch.masked_select(b_prediction, bmask)
                    mis_correct = torch.masked_select(is_correct, bmask)

                    pred_stats[mask_name].add_(mY, mb_prediction, mis_correct)
                    acc_stats[mask_name].add_(mY, mb_prediction, mis_correct)

            Logger.debug(
                "Accuracy: {}".format(
                    " | ".join(
                        "{:>12s} {}".format(key, stats)
                        for key, stats in acc_stats.items()
                    )
                )
            )
            if verbose and acc_stats["all"].get_accuracy() > 0:
                Logger.debug("Most common predictions:")
                for key, stats in pred_stats.items():
                    stats.dump_most_common(key, 10)

            res = {}
            for name in sorted(acc_stats.keys()):
                stats = acc_stats[name]
                res[name + "_correct"] = stats.num_correct
                res[name + "_samples"] = stats.num_samples
                res[name + "_acc"] = round(stats.get_accuracy(), 2)

            for key, stats in pred_stats.items():
                prec = stats.get_precision(include_reject=False)
                res[key + "_noreject_correct"] = prec.correct
                res[key + "_noreject_predicted"] = prec.predicted
                res[key + "_noreject_acc"] = prec.precision

            return res

    def accuracy(
        self,
        dataset_iter: Iterable[MiniBatch],
        out_field: torchtext.data.field.Field,
        verbose=True,
    ):
        return self.accuracy_with_reject(
            dataset_iter, out_field, reject_id=None, verbose=verbose
        )

    """
    Adversarial
    """

    def get_adversarial_mask(
        self,
        batch: MiniBatch,
        mask_field: Optional[str] = None,
        previous_mask: Optional[Tensor] = None,
    ):
        if previous_mask is None:
            # initialize to all predictable values
            previous_mask = self.padding_mask(batch, mask_field)

        self.predict_batch(batch)
        with torch.no_grad():
            h = self(batch.X)
            b_pred = self.softmax.predict(h)
            # We don't care about padding and unknown tokens, this should be already masked, but to be sure
            padding_mask = self.padding_mask(batch)
            is_correct = (b_pred == batch.Y) * padding_mask

        return (previous_mask * is_correct).detach()
