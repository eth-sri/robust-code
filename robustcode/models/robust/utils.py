import os

from torch import nn
from torch import optim

from robustcode.models.modules.neural_model_base import NeuralModelBase
from robustcode.models.robust.adversary.adversary import AdversarialMode
from robustcode.models.robust.adversary.adversary import AdversaryAccuracyStats
from robustcode.models.robust.adversary.adversary import AdversaryBatchIter
from robustcode.models.robust.adversary.adversary import RenameAdversary
from robustcode.models.robust.adversary.adversary import SubtreeAdversary
from robustcode.models.robust.adversary.rules import AdversarialNodeReplacement
from robustcode.models.robust.adversary.rules import NodeValueIndex
from robustcode.models.robust.adversary.rules import NodeValueIndexStr
from robustcode.models.robust.adversary.tree_rules import AdversarialSubtreeReplacement
from robustcode.models.robust.adversary.tree_rules import ExpressionGenerator
from robustcode.models.robust.dataset import Dataset
from robustcode.models.robust.dataset_util import dataset_to_trees
from robustcode.models.robust.dataset_util import dataset_to_trees_num
from robustcode.util.misc import Logger


def checkpoint_name(args, model_id):
    return "{}.pt".format(model_id)


def checkpoint_dir(args):
    name = "adv{}_val{}".format(args.adversarial, args.include_values)
    if args.window_size != 0:
        name += "_window{}".format(args.window_size)
    checkpoint_dir = os.path.join(args.save_dir, args.tag, name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def load_model(model: NeuralModelBase, args, model_id):
    import torch

    checkpoint_file = os.path.join(
        checkpoint_dir(args), checkpoint_name(args, model_id)
    )
    print("checkpoint_file", checkpoint_file)
    if not os.path.exists(checkpoint_file):
        return False

    Logger.debug("Loading model from {}".format(checkpoint_file))
    data = torch.load(checkpoint_file)
    model.load_state_dict(data)
    return True


def save_model(model: NeuralModelBase, args, model_id):
    import torch

    checkpoint_file = os.path.join(
        checkpoint_dir(args), checkpoint_name(args, model_id)
    )
    Logger.debug("Saving model to {}".format(checkpoint_file))
    torch.save(model.state_dict(), checkpoint_file)


def make_adversary(dataset: Dataset, make_iter):
    Logger.start_scope("Parsing Trees")
    trees_train_str = dataset_to_trees(dataset.dtrain, dataset.ID)
    trees_valid_str = dataset_to_trees(dataset.dvalid, dataset.ID)
    trees_test_str = dataset_to_trees(dataset.dtest, dataset.ID)
    trees_str = {**trees_train_str, **trees_valid_str, **trees_test_str}

    trees_train_num = dataset_to_trees_num(dataset.dtrain)
    trees_valid_num = dataset_to_trees_num(dataset.dvalid)
    trees_test_num = dataset_to_trees_num(dataset.dtest)
    trees_num = {**trees_train_num, **trees_valid_num, **trees_test_num}
    Logger.end_scope()

    Logger.start_scope("Indexing Trees")
    value_index = NodeValueIndex(dataset, trees_train_num)
    value_index_str = NodeValueIndexStr(dataset, trees_train_str)
    expr_gen = ExpressionGenerator(value_index_str)

    node_replacement = AdversarialNodeReplacement(
        value_index, dataset.fixed_value_offset
    )
    rules_index = node_replacement.make_rules(dataset, trees_str, trees_num)
    adversary = RenameAdversary(rules_index, dataset)
    Logger.end_scope()

    subtree_replacement = AdversarialSubtreeReplacement(expr_gen)
    subtree_rules = subtree_replacement.make_rules(dataset, trees_str, trees_num)
    subtree_adversary = SubtreeAdversary(subtree_rules, dataset, trees_str, make_iter)

    return adversary, subtree_adversary


def train_base_model(
    model: NeuralModelBase,
    dataset: Dataset,
    num_epochs,
    train_iter,
    valid_iter,
    lr=0.001,
    verbose=True,
):
    valid_iters = [valid_iter] if not isinstance(valid_iter, list) else valid_iter
    Logger.start_scope("Training Model")
    opt = optim.Adam(model.parameters(), lr=lr)
    model.opt = opt
    loss_function = nn.CrossEntropyLoss(reduction="none")
    model.loss_function = loss_function

    train_prec, valid_prec = None, None
    for epoch in range(num_epochs):
        Logger.start_scope("Epoch {}".format(epoch))
        model.fit(train_iter, opt, loss_function, mask_field="mask_valid")

        for valid_iter in valid_iters:
            valid_stats = model.accuracy(valid_iter, dataset.TARGET, verbose=verbose)
            valid_prec = valid_stats["mask_valid_noreject_acc"]
            Logger.debug(f"valid_prec: {valid_prec}")
        Logger.end_scope()

    train_stats = model.accuracy(train_iter, dataset.TARGET, verbose=False)
    train_prec = train_stats["mask_valid_noreject_acc"]
    Logger.debug(f"train_prec: {train_prec}, valid_prec: {valid_prec}")
    Logger.end_scope()
    return train_prec, valid_prec


def eval_adversarial(
    model: NeuralModelBase,
    it,
    rename_adversary: RenameAdversary,
    subtree_adversary: SubtreeAdversary,
    n_renames=20,
    n_subtree_renames=50,
    adv_mode=AdversarialMode.RANDOM,
    threshold=0.5,
    out_file=None,
    approximate=False,
) -> AdversaryAccuracyStats:
    Logger.debug(
        "Eval Adversarial [n_renames={}, n_subtree={}, mode={}]".format(
            n_renames, n_subtree_renames, adv_mode
        )
    )
    iterators = []
    if n_renames == 0:
        assert adv_mode == AdversarialMode.RANDOM
        iterators.append(
            AdversaryBatchIter(
                subtree_adversary, model, it, num_samples=n_subtree_renames
            )
        )
    else:
        if n_renames > 0:
            iterators.append(
                AdversaryBatchIter(
                    rename_adversary,
                    model,
                    it,
                    num_samples=n_renames,
                    adv_mode=adv_mode,
                )
            )

        if n_subtree_renames > 0:
            if n_renames == 0:
                iterators.append(
                    AdversaryBatchIter(
                        subtree_adversary, model, it, num_samples=n_subtree_renames
                    )
                )
            else:
                iterators.append(
                    AdversaryBatchIter(
                        subtree_adversary,
                        model,
                        AdversaryBatchIter(
                            rename_adversary,
                            model,
                            it,
                            num_samples=n_subtree_renames,
                            adv_mode=adv_mode,
                        ),
                    )
                )

    return rename_adversary.adversarial_accuracy(
        model,
        it,
        iterators,
        threshold=threshold,
        out_file=out_file,
        verbose=True,
        approximate=approximate,
    )
