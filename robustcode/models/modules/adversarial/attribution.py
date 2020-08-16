import torch

from robustcode.models.modules.dgl.utransformer import GraphModel
from robustcode.models.modules.iterators import MiniBatch
from robustcode.models.modules.neural_model_base import DisableDropout
from robustcode.models.modules.neural_model_base import NeuralModelBase


class AttributionHelper:
    def __init__(self, base_model: NeuralModelBase):
        self.base = base_model

        # # used for debug printing
        # self.trees_str = trees_str
        # self.target_field = target_field

    """
        Computes gradient for all the positions in the input for which mask == 1

        Parameters
        ----------
        batch: MiniBatch

        loss_function:


        Returns
        -------
        all_indices: Tensor (BatchSize x InputLength),
            zero-padded input positions for which gradients were computed
        all_grads: List[Tensor (InputLength x BatchSize)]
            l-1 norm of the gradient computed for positions in all_indices           
    """

    def compute_batch_grads(
        self,
        batch: MiniBatch,
        batch_mask: torch.Tensor,
        loss_function,
        max_samples=None,
    ):
        self.base.train()
        with DisableDropout(self.base):
            all_grads, all_indices = self.compute_grads(
                batch, batch_mask, loss_function, max_samples
            )
            return all_grads, all_indices

    def compute_grads(
        self, mini_batch: MiniBatch, mask: torch.Tensor, loss_function, max_samples
    ):
        if isinstance(self.base, GraphModel):
            grads, indices = self.__compute_grads_graph(
                mini_batch, mask, loss_function, max_samples=max_samples
            )
            return grads, indices
        else:
            embed = self.base.embed(mini_batch.X)
            delta = torch.zeros_like(embed, requires_grad=True)
            out = self.base.forward_with_embed(embed + delta, mini_batch.X)
            loss = self.base.softmax_loss(out, mini_batch, loss_function=loss_function)

            grads, indices = self.__compute_grads(
                delta,
                mask,
                loss,
                self.base.padding_mask(mini_batch),
                max_samples=max_samples,
            )
            return grads, indices

    def __compute_grads_graph(
        self, mini_batch: MiniBatch, mask: torch.Tensor, loss_function, max_samples
    ):
        # arrange loss indices such that each row contains one value from each of the batches
        # this way they can be computed together without affecting each other
        assert isinstance(self.base, GraphModel)

        tree_sizes = mini_batch.lengths
        tree_masks = torch.split(mask, tree_sizes)

        if max_samples is None:
            num_samples = max(torch.sum(tree_mask).item() for tree_mask in tree_masks)
        else:
            num_samples = min(
                max_samples,
                max(torch.sum(tree_mask).item() for tree_mask in tree_masks),
            )
        # arrange loss indices such that each row contains one value from each of the trees
        # this way they can be computed together without affecting each other
        loss_indices = torch.zeros(
            len(tree_masks), num_samples, device=mask.device, dtype=torch.long
        )
        # initialize with -1 to denote that no element is present
        nonzero_indices = torch.zeros_like(loss_indices) + -1

        offset = 0
        for idx, tree_mask in enumerate(tree_masks):
            indices = torch.nonzero(tree_mask).squeeze(dim=1)
            if indices.numel() > num_samples:
                indices = indices[torch.randperm(indices.numel())[:num_samples]]
            loss_indices[idx, 0 : len(indices)] = indices + offset
            nonzero_indices[idx, 0 : len(indices)] = indices
            offset += tree_mask.numel()
        loss_indices = loss_indices.t().contiguous()

        # compute gradients
        grads = []
        for row_indices in loss_indices:
            # forward
            embed = self.base.embed(mini_batch.X)
            delta = torch.zeros_like(embed, requires_grad=True)
            out = self.base.forward_with_embed(embed + delta, mini_batch.X)
            loss = self.base.softmax_loss(out, mini_batch, loss_function=loss_function)

            # remove padding
            row_indices = row_indices[torch.nonzero(row_indices).squeeze(dim=1)]

            row_loss = torch.gather(loss, 0, row_indices)
            row_loss = row_loss.mean()

            # backward
            self.base.zero_grad()
            row_loss.backward()

            delta.grad.detach_()
            grad = torch.split(delta.grad, tree_sizes)
            grad = [g.norm(p=1, dim=-1) for g in grad]
            grad = [g / torch.sum(g, dim=0) for g in grad]
            grads.append(grad)
            delta.grad.zero_()

        return grads, nonzero_indices

    def __compute_grads(self, delta, mask, loss, padding_mask, max_samples):
        # arrange loss indices such that each row contains one value from each of the batches
        # this way they can be computed together without affecting each other
        batch_size = mask.size(dim=1)
        if max_samples is None:
            num_samples = torch.max(torch.sum(mask, dim=0))
        else:
            num_samples = min(max_samples, torch.max(torch.sum(mask, dim=0)))
        loss_indices = (
            torch.zeros(batch_size, num_samples, device=mask.device, dtype=torch.long)
            + -1
        )
        # initialize with -1 to denote that no element is present
        nonzero_indices = torch.zeros_like(loss_indices) + -1
        for idx, sample in enumerate(mask.t()):
            indices = torch.nonzero(sample).squeeze(dim=1)
            if indices.numel() > num_samples:
                indices = indices[torch.randperm(indices.numel())[:num_samples]]
            loss_indices[idx, 0 : len(indices)] = indices * batch_size + idx
            nonzero_indices[idx, 0 : len(indices)] = indices
        loss_indices = loss_indices.t().contiguous()
        # assert torch.sum(torch.nonzero(mask.view(-1))).item() == torch.sum(loss_indices.view(-1)).item()

        # compute gradients
        grads = []
        for row_indices in loss_indices:
            # remove padding
            row_indices = row_indices[row_indices >= 0]
            row_loss = torch.gather(loss, 0, row_indices)
            row_loss = row_loss.mean()

            # backward
            self.base.zero_grad()
            row_loss.backward(retain_graph=True)

            delta.grad.detach_()
            grad = delta.grad.norm(p=1, dim=-1)
            grad = grad * padding_mask
            grad = grad / torch.sum(grad, dim=0)
            grads.append(grad.t())
            delta.grad.zero_()

        return grads, nonzero_indices
