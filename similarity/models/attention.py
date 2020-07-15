import torch
import torch.nn as nn
from typing import Tuple

from utils.parsing import Arguments
from utils.utils import compute_cost, prod


def load_attention_layer(args: Arguments, bidirectional: bool) -> nn.Module:
    if args.attention_type == 0 or args.attention_type == 1:
        return attention0(args, bidirectional)
    if args.attention_type == 2 or args.attention_type == 3:
        return attention2(args, bidirectional)
    if args.attention_type == 4:
        return attention4(args, bidirectional)
    if args.attention_type == 5:
        return attention5(args, bidirectional)


def build_ffn(input_size: int, output_size: int, args: Arguments) -> nn.Module:
    """Builds a 2-layer feed-forward network."""
    return nn.Sequential(
        # nn.Linear(input_size, self.args.hidden_size),
        # nn.Dropout(self.args.dropout),
        # nn.ReLU(),
        # nn.Linear(self.args.hidden_size, output_size)
        nn.Linear(input_size, output_size),
        nn.Dropout(args.dropout),
        nn.LeakyReLU(0.2),
    )


class attention5(nn.Module):
    def __init__(self, args: Arguments, input_size: int):
        super(attention5, self).__init__()
        self.args = args
        self.G = build_ffn(
            input_size=input_size, output_size=self.args.hidden_size, args=self.args
        )
        self.sparsemax = SparsemaxFunction.apply
        device = (
            torch.device(args.gpu) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = device

    def forward(
        self,
        row_vecs: torch.FloatTensor,
        column_vecs: torch.FloatTensor,
        cost: torch.FloatTensor,
        threshold: float = 0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # https://arxiv.org/abs/1606.01933
        # Attend
        a, b = row_vecs, column_vecs  # n x hidden_size, m x hidden_size

        if self.args.using_sparsemax:
            row_alignment = self.sparsemax(-cost, 1)
            column_alignment = self.sparsemax(-cost, 0)
        else:
            row_alignment = (-cost).softmax(dim=1)
            column_alignment = (-cost).softmax(dim=0)

        # if threshold:
        #     mask = (torch.rand(row_alignment.size(0), row_alignment.size(1)) >= threshold ).float().cuda()
        #     # threshold_alignments = [alignment * ( torch.rand(alignment.size(-2), alignment.size(-1)) >=0.5 ).float() for alignment in alignments]

        #     row_alignment = row_alignment * mask #(row_alignment >= threshold/size).float()
        #     column_alignment = column_alignment * mask #(column_alignment >= threshold/size).float()

        if threshold:
            # if self.args.absolute_threshold:
            #     alignment_masks_r = (row_alignments >= threshold).float().to(self.device)
            #     alignment_masks_c = (column_alignments >= threshold).float().to(self.device)
            # else:
            # mask = (torch.rand(row_alignments.size(0), row_alignments.size(1), row_alignments.size(2)) >= threshold ).float().to(self.device)
            # threshold_alignments = [alignment * ( torch.rand(alignment.size(-2), alignment.size(-1)) >=0.5 ).float() for alignment in alignments]
            # print('relative')
            alignment_masks_r = (
                (row_alignment >= threshold / prod(row_alignment.shape[-2:]))
                .float()
                .to(self.device)
            )
            alignment_masks_c = (
                (column_alignment >= threshold / prod(column_alignment.shape[-2:]))
                .float()
                .to(self.device)
            )

            row_alignment = (
                row_alignment * alignment_masks_r * alignment_masks_c
            )  # (row_alignment >= threshold/size).float()
            column_alignment = (
                column_alignment * alignment_masks_c * alignment_masks_r
            )  # (column_alignment >= threshold/size).float()

        beta = torch.sum(
            row_alignment.unsqueeze(dim=2) * b.unsqueeze(dim=0), dim=1
        )  # n x hidden_size
        alpha = torch.sum(
            column_alignment.unsqueeze(dim=2) * a.unsqueeze(dim=1), dim=0
        )  # m x hidden_size

        # Compare
        if self.args.force_attention_linear:
            v_1 = beta.mean(dim=0, keepdim=True)  # n x hidden_size
            v_2 = alpha.mean(dim=0, keepdim=True)  # m x hidden_size
        else:
            v_1 = self.G(beta).mean(dim=0, keepdim=True)  # n x hidden_size
            v_2 = self.G(alpha).mean(dim=0, keepdim=True)  # m x hidden_size

        y = compute_cost(cost_fn=self.args.cost_fn, x1=v_1, x2=v_2)

        cost_matrix = y * torch.ones_like(cost)
        alignment = torch.stack(
            (row_alignment.detach(), column_alignment.detach()), dim=0
        )
        return cost_matrix, alignment


"""Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""
"""
An implementation of sparsemax (Martins & Astudillo, 2016). See
:cite:`DBLP:journals/corr/MartinsA16` for detailed description.
By Ben Peters and Vlad Niculae
"""
# From: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/sparse_activations.py

import torch
from torch.autograd import Function
import torch.nn as nn


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold
    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None
