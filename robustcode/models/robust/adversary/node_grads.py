import collections
import itertools

import dgl
import networkx as nx
import torch
import tqdm

from robustcode.models.modules.adversarial.attribution import AttributionHelper
from robustcode.models.modules.dgl.utransformer import GraphModel
from robustcode.models.modules.iterators import MiniBatch
from robustcode.models.modules.neural_model_base import NeuralModelBase
from robustcode.models.robust.visualization import FieldLabel
from robustcode.models.robust.visualization import GradsLabel
from robustcode.models.robust.visualization import MaskLabel
from robustcode.models.robust.visualization import PredictLabel
from robustcode.models.robust.visualization import TopkLabel
from robustcode.models.robust.visualization import TreeVisualization


def each_node_grads(
    it, model: NeuralModelBase, dataset, loss_function, threshold=0.5, max_samples=None
):
    for batch in tqdm.tqdm(it, ncols=100, leave=False):
        yield from each_node_grads_batch(
            batch,
            model,
            dataset,
            loss_function,
            threshold=threshold,
            max_samples=max_samples,
        )


NodeGrads = collections.namedtuple(
    "NodeGrads",
    [
        "src_node",
        "probs",
        "tgt_nodes",
        "tree_nx",
        "rev_tree_nx",
        "tree",
        "pred",
        "target",
        "step",
        "sample",
        "values",
    ],
)


def each_node_grads_batch(
    batch: MiniBatch,
    model: NeuralModelBase,
    dataset,
    loss_function,
    threshold=0.5,
    verbose=False,
    max_samples=None,
    masks=None,
):
    attr_helper = AttributionHelper(model)

    mask_field = "mask_valid"
    preds = model.predict_with_reject(
        batch, dataset.reject_token_id, threshold=threshold
    )
    targets = batch.Y  # list(model.ground_truth_batch(batch))

    t_mask = model.padding_mask(batch, mask_field=mask_field)
    if masks is None:
        # compute the gradients only for positions that were not rejected
        p_mask = preds != dataset.reject_token_id
        mask = t_mask & p_mask
    else:
        # masks after filtering only mask_valid samples
        mask_all = torch.zeros_like(t_mask)
        mask_all[t_mask] = masks
        # TODO: equivalent to t_mask & masks?
        mask = mask_all

    all_grads, all_indices = attr_helper.compute_batch_grads(
        batch, mask, loss_function, max_samples=max_samples
    )

    if not all_grads:
        return

    samples = dataset.samples_for_batch(batch)
    if isinstance(model, GraphModel):
        g = batch.X
        assert "preds" not in g.ndata
        g.ndata["preds"] = preds
        trees = dgl.unbatch(g)
        del g.ndata["preds"]

        for tree_id, sample, tree in zip(itertools.count(), samples, trees):
            grads_label = GradsLabel.from_values(all_grads, all_indices, tree_id)
            if verbose:
                TreeVisualization.visualize(
                    sample,
                    ["types", "values"],
                    dataset.dtrain.fields,
                    labels=[
                        FieldLabel(dataset.dtrain.fields["types"], tree.ndata["types"]),
                        FieldLabel(
                            dataset.dtrain.fields["values"], tree.ndata["values"]
                        ),
                        PredictLabel.from_iter(
                            [batch],
                            sample,
                            model,
                            dataset,
                            mask=MaskLabel.from_sample(sample, mask_field),
                        ),
                        TopkLabel.from_iter(
                            [batch],
                            sample,
                            model,
                            dataset,
                            mask=MaskLabel.from_sample(sample, mask_field),
                        ),
                        grads_label,
                    ],
                )

            tree_nx = tree.to_networkx()
            """
            reverse edge direction.
            during training/inference edges flow _into_ the source node.
            during edge filtering, we need to compute reachable nodes _from_ the source node
            which is easier if the direction is reversed
            """
            rev_tree_nx = nx.reverse_view(tree_nx)

            tree_preds = tree.ndata["preds"].cpu()
            for src_node, probs, tgt_nodes in grads_label:
                yield NodeGrads(
                    src_node,
                    probs,
                    tgt_nodes,
                    tree_nx,
                    rev_tree_nx,
                    tree,
                    tree_preds[src_node],
                    batch.Y[src_node],
                    None,
                    sample,
                    tree.ndata["values"],
                )
    else:
        bptt_len = preds[0].size(dim=0)
        for tree_id, sample in zip(itertools.count(), samples):
            grads_label = GradsLabel.from_values(all_grads, all_indices, tree_id)
            values = batch.inputs[-1][:, tree_id]
            if verbose:
                TreeVisualization.visualize(
                    sample,
                    ["types", "values"],
                    dataset.dtrain.fields,
                    labels=[
                        FieldLabel(dataset.dtrain.fields["values"], values),
                        PredictLabel.from_iter(
                            [batch],
                            sample,
                            model,
                            dataset,
                            mask=MaskLabel.from_sample(sample, "mask_valid"),
                        ),
                        TopkLabel.from_iter(
                            [batch],
                            sample,
                            model,
                            dataset,
                            mask=MaskLabel.from_sample(sample, "mask_valid"),
                        ),
                        grads_label,
                    ],
                )

            for src_node, probs, tgt_nodes in grads_label:
                mini_batch_id = src_node // bptt_len
                yield NodeGrads(
                    src_node,
                    probs,
                    tgt_nodes,
                    None,
                    None,
                    None,
                    preds[mini_batch_id][src_node % bptt_len, tree_id],
                    targets[mini_batch_id][src_node % bptt_len, tree_id],
                    None,
                    sample,
                    values,
                )

            if verbose:
                input()
