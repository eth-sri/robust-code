import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class RejectionCrossEntropyLoss(nn.Module):
    """
    Based on https://arxiv.org/pdf/1907.00208.pdf
    o (payoff) should be between 1 < o < num_classes
    """

    def __init__(self, o, out_vocab, rejection_index, reduction="none", weight=None):
        super(RejectionCrossEntropyLoss, self).__init__()
        assert 0 <= rejection_index < out_vocab
        assert reduction == "none", 'Only reduction="none" is currently supported'
        assert o is not None
        self.reduction = reduction
        self.rejection_index = rejection_index
        self.o = o
        self.softmax = nn.Softmax(dim=-1)
        self.weight = weight

    def forward(self, x, y):
        p = self.softmax(x)
        p_m = p.select(-1, self.rejection_index).unsqueeze(dim=1)
        loss = F.nll_loss(
            torch.log(p * self.o + p_m), y, reduction="none", weight=self.weight
        )
        return loss

    @staticmethod
    def args():
        parser = argparse.ArgumentParser("RejectionCrossEntropyLoss", add_help=False)
        parser.add_argument(
            "--o", type=float, help="Payoff value, should in range 1 < o < num_classes"
        )
        return parser
