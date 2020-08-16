from abc import ABC
from abc import abstractmethod
from typing import Iterable

import numpy
from sty import bg
from sty import rs

from robustcode.util.misc import marker
from robustcode.util.misc import trim
from robustcode.util.misc import word


class TreeLabel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def draw(self, idx):
        pass


class MaskLabel(TreeLabel):
    def __init__(self, values, color="da_black"):
        super(MaskLabel, self).__init__()
        self.values = values
        self.color = color

    def draw(self, idx):
        if self.values[idx]:
            return bg(self.color) + " " + rs.bg
        return " "

    @staticmethod
    def from_sample(sample, mask_name):
        assert hasattr(sample, mask_name)
        return MaskLabel(getattr(sample, mask_name))

    @staticmethod
    def from_values(values, color="da_black"):
        return MaskLabel(values, color=color)


class BooleanWithMaskLabel(TreeLabel):
    def __init__(self, mask, values):
        super(BooleanWithMaskLabel, self).__init__()
        assert len(mask) == len(values)
        self.mask = mask
        self.values = values

    def draw(self, idx):
        if self.mask[idx]:
            return marker(self.values[idx])
        return " "

    @staticmethod
    def from_values(mask, values):
        return BooleanWithMaskLabel(mask, values)


class TopkLabel(TreeLabel):
    def __init__(self, target_field, topk_probs, topk_targets, mask: MaskLabel = None):
        super(TopkLabel, self).__init__()
        self.target_field = target_field
        self.topk_probs = topk_probs
        self.topk_targets = topk_targets
        self.mask = mask

    def draw(self, idx):
        mask = self.mask.values[idx] if self.mask is not None else True
        if not mask:
            return " ".join("{:<15s}".format(""))
        topk_desc = " ".join(
            "{:.2f}:{:<10s}".format(p, trim(self.target_field.vocab.itos[t], 10))
            for p, t in zip(self.topk_probs[idx], self.topk_targets[idx])
        )
        return topk_desc

    @staticmethod
    def from_iter(it, sample, model, dataset, mask: MaskLabel = None):
        topk_probs, topk_targets = model.topk(
            it, k=2, id=dataset.ID.vocab.stoi[sample.id]
        )
        target_fields = [
            field for field in dataset.dtrain.fields.values() if field.is_target
        ]
        assert len(target_fields) == 1
        return TopkLabel(target_fields[0], topk_probs, topk_targets, mask)


class FieldLabel(TreeLabel):
    def __init__(self, field, values):
        super(FieldLabel, self).__init__()
        self.field = field
        self.values = values

    def draw(self, idx):
        return "{:>14s}".format(trim(self.field.vocab.itos[self.values[idx]], 14))


class PredictLabel(TreeLabel):
    def __init__(self, target_field, predictions, ground_truth, mask: MaskLabel = None):
        super(PredictLabel, self).__init__()
        self.target_field = target_field
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.mask = mask

    def draw(self, idx):
        mask = self.mask.values[idx] if self.mask is not None else True
        if not mask:
            return "  {:10s}".format("")
        prediction_label = self.target_field.vocab.itos[self.predictions[idx]]

        def make_marker(label, ground_truth):
            if label == "reject":
                return bg("da_black") + " " + rs.bg
            return marker(label == ground_truth)

        return "{} {:10s}".format(
            make_marker(prediction_label, self.ground_truth[idx]),
            trim(prediction_label, 10),
        )

    @staticmethod
    def from_iter(it, sample, model, dataset, mask: MaskLabel = None):
        predictions = model.predict(it, id=dataset.ID.vocab.stoi[sample.id])
        return PredictLabel.from_values(predictions, sample, dataset, mask)

    @staticmethod
    def from_values(predictions, sample, dataset, mask: MaskLabel = None):
        target_fields = [
            (name, field)
            for name, field in dataset.dtrain.fields.items()
            if field.is_target
        ]
        assert len(target_fields) == 1
        field_name, target_field = target_fields[0]
        ground_truth = getattr(sample, field_name)
        return PredictLabel(target_field, predictions, ground_truth, mask)


class GradsLabel(TreeLabel):
    def __init__(self, grads, indices, pos):
        super(GradsLabel, self).__init__()
        self.grads = grads
        # indices of nodes for which gradients were computed
        self.indices = indices[pos].cpu().numpy()
        self.batch_pos = pos

        self.bppt_len = self.grads[0][pos].size(dim=0)

    def draw(self, idx):
        if idx not in self.indices:
            return ""

        idx_pos = numpy.flatnonzero(self.indices == idx)[0]
        probs, preds = self.grads[idx_pos][self.batch_pos].topk(k=5, dim=-1)

        offset = int(idx // self.bppt_len) * self.bppt_len
        return " ".join(
            "{:.2f}:{:<4d}".format(p, t + offset)
            for p, t in zip(probs, preds)
            if p > 0.01
        )

    def __iter__(self):
        for idx_pos, node_idx in enumerate(self.indices):
            if node_idx == -1:  # -1 corresponds to a padded element
                # mini_batches are appended to each other, therefore there can be gaps (-1) in between
                continue
            probs, preds = self.grads[idx_pos][self.batch_pos].topk(k=5, dim=-1)
            preds.add_(int(node_idx // self.bppt_len) * self.bppt_len)
            yield node_idx, probs, preds

    @staticmethod
    def from_iter(it, sample, model, dataset, mask_field, attr_helper, loss_function):
        batch, pos = dataset.find_batch_for_sample(it, sample)

        all_grads, all_indices = attr_helper.compute_batch_grads(
            batch,
            lambda mini_batch, mini_batch_id: model.training_mask(
                mini_batch, mask_field=mask_field
            ),
            loss_function,
        )

        return GradsLabel(all_grads, all_indices, pos)

    @staticmethod
    def from_values(all_grads, all_indices, pos):
        return GradsLabel(all_grads, all_indices, pos)


class TreeVisualization:
    @staticmethod
    def visualize(sample, input_fields, fields, labels: Iterable[TreeLabel] = None):
        labels = labels or []

        target_fields = [name for name, field in fields.items() if field.is_target]
        field_names = input_fields + target_fields

        field_values = [getattr(sample, name) for name in field_names]
        length = len(field_values[0])
        assert all(length == len(v) for v in field_values)

        depth = getattr(sample, "depth") if hasattr(sample, "depth") else [0] * length
        for idx in range(length):
            indent = "{:<{}s}{:4d}".format("=" * depth[idx], max(depth), idx)
            values = "\t".join(
                word(v[idx], fields[name]) for v, name in zip(field_values, field_names)
            )

            print(indent, values, " ".join(label.draw(idx) for label in labels))
