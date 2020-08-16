import functools
import os
import sys

import pandas as pd
import torch

from robustcode.models.modules.dgl.gcn import GCNNet
from robustcode.models.modules.dgl.ggnn import GGNNNet
from robustcode.models.modules.dgl.utransformer import GraphModel
from robustcode.models.modules.dgl.utransformer import UGraphTransformer
from robustcode.models.modules.rejection_cross_entropy_loss import (
    RejectionCrossEntropyLoss,
)
from robustcode.models.modules.rnn.model import RNNWithAttention
from robustcode.models.modules.util import Random
from robustcode.models.robust.adversary.adversary import AdversarialMode
from robustcode.models.robust.adversary.adversary import AdversaryBatchIter
from robustcode.models.robust.adversary.adversary import RenameAdversary
from robustcode.models.robust.adversary.adversary import SubtreeAdversary
from robustcode.models.robust.dataset import Dataset
from robustcode.models.robust.dataset import Iterators
from robustcode.models.robust.dataset import Models
from robustcode.models.robust.utils import checkpoint_dir
from robustcode.models.robust.utils import eval_adversarial
from robustcode.models.robust.utils import load_model
from robustcode.models.robust.utils import make_adversary
from robustcode.models.robust.utils import save_model
from robustcode.models.robust.utils import train_base_model
from robustcode.util.argparse import ArgConfigParser
from robustcode.util.misc import boolean_string
from robustcode.util.misc import Logger


def parse_args():
    parser = ArgConfigParser(
        "Robust Code Standard + Adversarial Training (without Abstain)",
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


def baseline(
    args,
    dataset: Dataset,
    device: torch.device,
    train_iter,
    valid_iter,
    test_iter,
    rename_adversary: RenameAdversary,
    subtree_adversary: SubtreeAdversary,
    train_adversarial=False,
    model_id=None,
):
    model = Models.make(args, dataset, device, train_iter)
    model.print_info()

    if train_adversarial:
        if args.n_renames == 0:
            train_iter = AdversaryBatchIter(
                subtree_adversary, model, train_iter, num_samples=1
            )
        else:
            train_iter = AdversaryBatchIter(
                subtree_adversary,
                model,
                AdversaryBatchIter(
                    rename_adversary,
                    model,
                    train_iter,
                    num_samples=1,
                    adv_mode=args.train_adv_mode,
                ),
            )

    if model_id is None or not load_model(model, args, model_id):
        train_base_model(model, dataset, args.num_epochs, train_iter, valid_iter)

        if model_id is not None:
            save_model(model, args, model_id)

    Logger.debug("Valid Accuracy")
    valid_acc = model.accuracy(valid_iter, dataset.TARGET)
    Logger.debug("Test Accuracy")
    test_acc = model.accuracy(test_iter, dataset.TARGET)

    Random.seed(42)
    test_adv_stats = eval_adversarial(
        model,
        test_iter,
        rename_adversary,
        subtree_adversary,
        n_renames=args.n_renames,
        n_subtree_renames=args.n_subtree,
        adv_mode=args.adv_mode,
        out_file=args.out_file,
        approximate=True,
    )
    Logger.debug("Valid Accuracy")
    print(valid_acc)
    Logger.debug("Test Accuracy")
    print(test_acc)
    d = test_adv_stats.to_dict()
    d["valid_acc"] = valid_acc["mask_valid_acc"]
    d["test_acc"] = test_acc["mask_valid_acc"]
    d = {k: [v] for k, v in d.items()}
    return pd.DataFrame(data=d, index=[model_id])


def main():
    args = parse_args()
    if not args.include_values:
        # When the values are not included renaming is a no-op
        args.n_renames = 0
    if args.adv_mode != "RANDOM" or args.train_adv_mode != "RANDOM":
        args.dot_product_embedding = True

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

    """
    Model
    """
    masks = {"mask_valid": dataset.MASK_VALID}
    train_iter, valid_iter, test_iter = Iterators.make(
        args, Models[args.model], dataset, device, masks
    )

    # visualize_dataset(train_iter, dataset)

    """
    Training
    """
    rename_adversary, subtree_adversary = make_adversary(
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
        df = baseline(
            args,
            dataset,
            device,
            train_iter,
            valid_iter,
            test_iter,
            rename_adversary,
            subtree_adversary,
            train_adversarial=args.adversarial,
            model_id=i,
        )
        dfs.append(df)
        save_results(dfs)


if __name__ == "__main__":
    main()
