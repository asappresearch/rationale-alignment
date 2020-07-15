from functools import partial
import math
from typing import Any, Iterator, List, Optional, Tuple

from sinkhorn import batch_sinkhorn, construct_cost_and_marginals
from sru import SRU
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.parsing import Arguments
from utils.utils import compute_cost, prod, unpad_tensors, feed_forward
from classify.models.attention import load_attention_layer, SparsemaxFunction
from classify.models.encoder import Embedder


class AlignmentModel(nn.Module):
    def __init__(
        self,
        args: Arguments,
        text_field,
        domain: str = "",
        device: torch.device = torch.device("cpu"),
    ):
        """Constructs an AlignmentModel."""
        super(AlignmentModel, self).__init__()

        # Save values
        self.args = args
        self.device = device

        self.embedder = Embedder(
            args=args,
            text_field=text_field,
            pooling=self.args.embedder_pooling,
            device=device,
        )

        if self.args.alignment == "average":
            self.args.attend_l = self.args.compare_l = 0

        self.attention_forward = feed_forward(
            self.embedder.output_size,
            self.args.ffn_hidden_size,
            self.args.dropout,
            self.args.attend_l,
            self.args.attend_activation,
        )

        compare_input = (
            2 * self.args.ffn_hidden_size
            if self.args.attend_l
            else 2 * self.embedder.output_size
        )
        self.compare_forward = feed_forward(
            compare_input,
            self.args.ffn_hidden_size,
            self.args.dropout,
            self.args.compare_l,
        )

        aggregate_input = (
            2 * self.args.ffn_hidden_size if self.args.compare_l else compare_input
        )
        self.cls_forward = feed_forward(
            aggregate_input,
            self.args.ffn_hidden_size,
            self.args.dropout,
            self.args.aggregate_l,
        )

        cls_input = (
            self.args.ffn_hidden_size if self.args.aggregate_l else aggregate_input
        )

        self.domain = domain
        if self.domain == "snli":
            self.out = nn.Linear(cls_input, 3)
            self.word_to_word = True
        if self.domain == "multirc":
            self.out = nn.Linear(cls_input, 2)

        # This is a special model for SNLI, where the logits is (pos_cost, neg_cost, bias)
        self.word_to_word = True
        self.pad_index = text_field.pad_index()
        # self.snli = domain=='snli'
        self.output_size = self.embedder.output_size
        self.sparsemax = SparsemaxFunction.apply

        self.to(self.device)

    def forward(
        self,
        data: torch.LongTensor,  # batch_size x seq_len
        scope: List[Tuple[torch.LongTensor, torch.LongTensor]],
        targets: List[torch.LongTensor],
        threshold: float = 0,
        epsilon: float = 1e-4,
    ) -> Tuple[List[Tuple[torch.FloatTensor, torch.FloatTensor]], List[Any]]:
        """
        Aligns document pairs.

        :param data: Sentences represented as LongTensors of word indices.
        :param scope: A list of tuples of row_indices and column_indices indexing into data
        to extract the appropriate sentences for each document pair.
        :param targets: A list of targets for each document pair.
        :return: A tuple consisting of a list of (cost, alignment) tuples and a list of targets.
        """
        encoded = self.embedder(
            data, return_sequence=True
        )  # bs(of sent), seq_len, n_hidden
        encoded = self.attention_forward(encoded)
        # print(encoded.shape)
        mask = (data != self.pad_index).float()  # bs, seq_len
        # if self.word_to_word:
        #     encoded = encodede_seq

        # Alignment
        costs, alignments = [], []
        logits = []
        n_list, m_list = [], []
        all_rows, all_columns = [], []
        row_idx = torch.cat(([s[0] for s in scope]), dim=0)
        column_idx = torch.cat(([s[1] for s in scope]), dim=0)

        n_max = torch.index_select(mask, 0, row_idx).sum(-1).max().long().item()
        rows = torch.index_select(encoded, 0, row_idx)[
            :, :n_max, :
        ]  # bs x (n/m)x 2*hidden_size
        row_mask = torch.index_select(mask, 0, row_idx)[:, :n_max]  # bs x n
        n_list = list(torch.sum(row_mask, dim=-1).long().cpu().numpy())

        m_max = torch.index_select(mask, 0, column_idx).sum(-1).max().long().item()
        columns = torch.index_select(encoded, 0, column_idx)[
            :, :m_max, :
        ]  # bs x (n/m)x 2*hidden_size
        column_mask = torch.index_select(mask, 0, column_idx)[:, :m_max]  # bs x m
        m_list = list(torch.sum(column_mask, dim=-1).long().cpu().numpy())

        if self.args.alignment == "average":
            row_avg = rows.sum(dim=1) / row_mask.sum(dim=1).unsqueeze(
                dim=-1
            )  # batch, hidden
            column_avg = columns.sum(dim=1) / column_mask.sum(dim=1).unsqueeze(dim=-1)
            logits = self.out(
                self.cls_forward(torch.cat((row_avg, column_avg), dim=-1))
            )
            costs = (
                compute_cost(
                    cost_fn=self.args.cost_fn,
                    x1=row_avg.unsqueeze(1),
                    x2=column_avg.unsqueeze(1),
                    batch=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            alignments = [
                torch.ones(n, m) / n * m for i, (n, m) in enumerate(zip(n_list, m_list))
            ]
            costs = [
                torch.ones(n, m) * costs[i]
                for i, (n, m) in enumerate(zip(n_list, m_list))
            ]
            preds = list(zip(costs, alignments))
            return preds, targets, logits

        batch_cost = compute_cost(
            cost_fn=self.args.cost_fn, x1=rows, x2=columns, batch=True
        )
        batch_cost += self.args.shiftcost

        if self.args.alignment == "sinkhorn":
            costs = [
                batch_cost[i, :n, :m] for i, (n, m) in enumerate(zip(n_list, m_list))
            ]
            # Alignment via sinkhorn
            # self.args.alignment == 'sinkhorn':
            # Add dummy node and get marginals
            costs, a_list, b_list = zip(
                *[
                    construct_cost_and_marginals(
                        C=cost,
                        one_to_k=self.args.one_to_k,
                        split_dummy=self.args.split_dummy,
                        max_num_aligned=self.args.max_num_aligned,
                        optional_alignment=self.args.optional_alignment,
                    )
                    for cost in costs
                ]
            )
            costs, a_list, b_list = list(costs), list(a_list), list(b_list)

            # Prepare sinkhorn function
            batch_sinkhorn_func = partial(
                batch_sinkhorn,
                a_list=a_list,
                b_list=b_list,
                epsilon=epsilon,
                unbalanced_lambda=self.args.unbalanced_lambda,
                error_check_frequency=self.args.error_check_frequency,
                error_threshold=self.args.error_threshold,
            )

            # Run sinkhorn
            alignments = batch_sinkhorn_func(
                C_list=costs, return_tensor=True
            )  # bs, n, m

            if threshold:
                if self.args.absolute_threshold:
                    alignments = alignments * (alignments >= threshold).float().to(
                        self.device
                    )
                elif self.args.ratio_threshold:

                    threshold_adjusted = b.sort()[0][int(b.shape[0] * threshold) - 1]
                    alignments = alignments * (
                        alignments >= threshold_adjusted
                    ).float().to(self.device)
                else:
                    alignment_masks = (
                        torch.stack(
                            [
                                alignments[i] >= threshold / (n * m)
                                for i, (n, m) in enumerate(zip(n_list, m_list))
                            ]
                        )
                        .float()
                        .to(self.device)
                    )
                    alignments = alignments * alignment_masks

            if alignments.shape[-2] != n_max or alignments.shape[-1] != m_max:
                # print('reshaping')
                alignments = torch.stack(
                    [alignment[:n_max, :m_max] for alignment in alignments]
                )

            column_alignments = alignments.transpose(1, 2)
            row_alignments = alignments

        elif self.args.alignment == "attention":
            # batch_cost, shape (bs, n, m)
            if self.args.using_sparsemax:
                row_alignments = self.sparsemax(
                    -batch_cost * row_mask.unsqueeze(2), 2
                )  # (bs, n, *) *
                column_alignments = self.sparsemax(
                    -batch_cost * column_mask.unsqueeze(1), 1
                ).transpose(1, 2)
            else:
                row_alignments = (-batch_cost * row_mask.unsqueeze(2)).softmax(
                    dim=2
                )  # (bs, n, *)
                column_alignments = (
                    (-batch_cost * column_mask.unsqueeze(1))
                    .softmax(dim=1)
                    .transpose(1, 2)
                )  # (bs, *, m)

            if threshold:
                if self.args.absolute_threshold:
                    alignment_masks_r = (
                        (row_alignments >= threshold).float().to(self.device)
                    )
                    alignment_masks_c = (
                        (column_alignments >= threshold).float().to(self.device)
                    )
                elif self.args.ratio_threshold:
                    b = row_alignments.reshape(-1)
                    threshold_adjusted_r = b.sort()[0][int(b.shape[0] * threshold) - 1]
                    alignment_masks_r = (
                        (row_alignments >= threshold_adjusted_r).float().to(self.device)
                    )
                    bb = column_alignments.reshape(-1)
                    threshold_adjusted_c = bb.sort()[0][
                        int(bb.shape[0] * threshold) - 1
                    ]
                    alignment_masks_c = (
                        (column_alignments >= threshold_adjusted_c)
                        .float()
                        .to(self.device)
                    )
                else:
                    alignment_masks_r = (
                        torch.stack(
                            [
                                row_alignments[i] >= threshold / (n * m)
                                for i, (n, m) in enumerate(zip(n_list, m_list))
                            ]
                        )
                        .float()
                        .to(self.device)
                    )
                    alignment_masks_c = (
                        torch.stack(
                            [
                                column_alignments[i] >= threshold / (n * m)
                                for i, (n, m) in enumerate(zip(n_list, m_list))
                            ]
                        )
                        .float()
                        .to(self.device)
                    )

                row_alignments = (
                    row_alignments
                    * alignment_masks_r
                    * alignment_masks_c.transpose(1, 2)
                )  # (row_alignment >= threshold/size).float()
                column_alignments = (
                    column_alignments
                    * alignment_masks_c
                    * alignment_masks_r.transpose(1, 2)
                )  # (column_alignment >= threshold/size).float()

        elif self.args.alignment == "all_pairs":
            alignments = torch.ones(len(rows), n_max, m_max).to(self.device)
            alignments = torch.stack(
                [alignments[i] / n * m for i, (n, m) in enumerate(zip(n_list, m_list))]
            )
            column_alignments = alignments.transpose(1, 2)
            row_alignments = alignments

        # weighted sum
        attended_row = column_alignments.contiguous().bmm(
            rows
        )  # (bs,m,n)*(bs,n,hidden) -> (bs,m,hidden)
        attended_column = row_alignments.bmm(
            columns
        )  # (bs,n,m)*(bs,m,hidden) -> (bs,n,hidden)
        # row_compare_input = torch.cat([embedded_row, attended_column], dim=-1)

        # pooling for the length: average pooling
        if self.args.compare_l == 0:
            compared_row = torch.sum(
                attended_column * row_mask.unsqueeze(-1), dim=1
            ) / row_mask.sum(1).unsqueeze(
                1
            )  # (bs,hidden)
            compared_column = torch.sum(
                attended_row * column_mask.unsqueeze(-1), dim=1
            ) / column_mask.sum(1).unsqueeze(
                1
            )  # (bs,hidden)
        else:
            row_compare_input = torch.cat([rows, attended_column], dim=-1)
            column_compare_input = torch.cat([columns, attended_row], dim=-1)

            compared_row = self.compare_forward(row_compare_input)
            compared_row = compared_row * row_mask.unsqueeze(-1)
            # Shape: (batch_size, compare_dim)
            compared_row = compared_row.sum(dim=1)

            compared_column = self.compare_forward(column_compare_input)
            compared_column = compared_column * column_mask.unsqueeze(-1)
            # Shape: (batch_size, compare_dim)
            compared_column = compared_column.sum(dim=1)

        logits = self.out(
            self.cls_forward(torch.cat((compared_row, compared_column), dim=-1))
        )

        if self.args.alignment != "attention":
            alignments = [
                alignments[i, :n, :m] for i, (n, m) in enumerate(zip(n_list, m_list))
            ]
        else:
            row_alignments = [
                row_alignments[i, :n, :m]
                for i, (n, m) in enumerate(zip(n_list, m_list))
            ]
            column_alignments = column_alignments.transpose(1, 2)
            column_alignments = [
                column_alignments[i, :n, :m]
                for i, (n, m) in enumerate(zip(n_list, m_list))
            ]
            alignments = [
                torch.stack(
                    (row_alignments[i].detach(), column_alignments[i].detach()), dim=0
                )
                for i in range(len(rows))
            ]
        costs = [batch_cost[i, :n, :m] for i, (n, m) in enumerate(zip(n_list, m_list))]

        # Re-normalize alignment probabilities to one
        if (
            self.args.alignment in ["attention", "sinkhorn"]
            and not self.args.optional_alignment
        ):
            alignments = [
                alignment
                / alignment.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
                for alignment in alignments
            ]

        # Combine costs and alignments into preds

        preds = list(zip(costs, alignments))

        return preds, targets, logits

    def forward_with_sentences(
        self,
        data: torch.LongTensor,
        scope: List[Tuple[torch.LongTensor, torch.LongTensor]],
        targets: List[torch.LongTensor],
        threshold: float = 0,
    ) -> Tuple[
        List[Tuple[torch.LongTensor, torch.LongTensor]],
        List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        List[Any],
    ]:
        """Makes predictions and returns input sentences, predictions, and targets."""
        sentences = [
            (
                torch.index_select(data, 0, simple_indices),
                torch.index_select(data, 0, normal_indices),
            )
            for simple_indices, normal_indices in scope
        ]
        preds, targets, _ = self.forward(data, scope, targets, threshold)

        return sentences, preds, targets

    def num_parameters(self, trainable: bool = False) -> int:
        """Gets the number of parameters in the model.
        Returns
        ----------
        int
            number of model params
        """
        if trainable:
            model_params = list(self.trainable_params)
        else:
            model_params = list(self.parameters())

        return sum(len(x.view(-1)) for x in model_params)

    @property
    def trainable_params(self) -> Iterator[nn.Parameter]:
        """Get all the parameters with `requires_grad=True`.
        Returns
        -------
        Iterator[nn.Parameter]
            Iterator over the parameters
        """
        return filter(lambda p: p.requires_grad, self.parameters())

    @property
    def gradient_norm(self) -> float:
        """Compute the average gradient norm.

        Returns
        -------
        float
            The current average gradient norm

        """
        # Only compute over parameters that are being trained
        parameters = filter(
            lambda p: p.requires_grad and p.grad is not None, self.parameters()
        )
        norm = math.sqrt(sum(param.grad.norm(p=2).item() ** 2 for param in parameters))

        return norm

    @property
    def parameter_norm(self) -> float:
        """Compute the average parameter norm.

        Returns
        -------
        float
            The current average parameter norm

        """
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        norm = math.sqrt(sum(param.norm(p=2).item() ** 2 for param in parameters))

        return norm
