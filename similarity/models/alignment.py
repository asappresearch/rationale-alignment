from functools import partial
import math
from typing import Any, Iterator, List, Optional, Tuple

from sinkhorn import batch_sinkhorn, construct_cost_and_marginals
from sru import SRU
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.parsing import Arguments
from utils.utils import compute_cost, prod, unpad_tensors
from similarity.models.attention import load_attention_layer
from similarity.models.encoder import Embedder


class AlignmentModel(nn.Module):
    def __init__(
        self,
        args: Arguments,
        text_field,
        domain: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Constructs an AlignmentModel."""
        super(AlignmentModel, self).__init__()

        # Save values
        self.args = args
        self.device = device
        self.embedder = Embedder(args=args, text_field=text_field, device=device)

        if self.args.alignment == "attention":
            self.atten = load_attention_layer(self.args, self.embedder.output_size)

        self.output_size = self.embedder.output_size
        # Move to device
        self.to(self.device)

    def forward(
        self,
        data: torch.LongTensor,  # batch_size x seq_len
        scope: List[Tuple[torch.LongTensor, torch.LongTensor]],
        targets: List[torch.LongTensor],
        threshold: float = 0,
        encoded: torch.Tensor = None,
    ) -> Tuple[List[Tuple[torch.FloatTensor, torch.FloatTensor]], List[Any]]:
        """
        Aligns document pairs.

        :param data: Sentences represented as LongTensors of word indices.
        :param scope: A list of tuples of row_indices and column_indices indexing into data
        to extract the appropriate sentences for each document pair.
        :param targets: A list of targets for each document pair.
        :return: A tuple consisting of a list of (cost, alignment) tuples and a list of targets.
        """
        if encoded is None:
            encoded, encoded_seq = self.embedder(data)
        # if self.word_to_word:
        #     encoded = encodede_seq

        # Alignment
        costs, alignments = [], []
        n_list, m_list = [], []
        for row_indices, column_indices in scope:
            # Select sentence vectors using indices
            row_vecs, column_vecs = (
                torch.index_select(encoded, 0, row_indices),
                torch.index_select(encoded, 0, column_indices),
            )  # (n/m)x 2*hidden_size

            # Get sizes
            n, m = len(row_vecs), len(column_vecs)
            n_list.append(n)
            m_list.append(m)

            # Average sentence embeddings
            if self.args.alignment == "average":
                row_vecs = row_vecs.mean(dim=0, keepdim=True)
                column_vecs = column_vecs.mean(dim=0, keepdim=True)

            # Compute cost
            cost = compute_cost(cost_fn=self.args.cost_fn, x1=row_vecs, x2=column_vecs)

            # Alignment-specific computation
            if self.args.alignment == "attention":
                cost, alignment = self.atten(
                    row_vecs, column_vecs, cost / self.args.attention_temp, threshold
                )
                alignments.append(alignment)

            # Add cost
            costs.append(cost)

            # Hack alignment matrix for models that don't do alignment
            if self.args.alignment not in ["attention", "sinkhorn"]:
                alignments.append(
                    torch.ones_like(cost) / prod(cost.shape)
                )  # use alignment to compute average cost

        # Alignment via sinkhorn
        if self.args.alignment == "sinkhorn":
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
                epsilon=self.args.epsilon,
                unbalanced_lambda=self.args.unbalanced_lambda,
            )

            # Run sinkhorn
            alignments = batch_sinkhorn_func(C_list=costs)

            # Remove dummy nodes
            shapes = list(zip(n_list, m_list))
            alignments = unpad_tensors(alignments, shapes)

            costs = unpad_tensors(costs, shapes)

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

        return preds, targets

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
        preds, targets = self.forward(data, scope, targets, threshold)

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
