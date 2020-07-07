from functools import partial
import math
from typing import Any, Iterator, List, Optional, Tuple

from sinkhorn import batch_sinkhorn, construct_cost_and_marginals
from sru import SRU
import torch
import torch.nn as nn
import torch.nn.functional as F

from rationale_alignment.parsing import Arguments
from rationale_alignment.utils import compute_cost, prod, unpad_tensors
from classify.models.attention import load_attention_layer
from classify.models.pooling_attention import (
    SelfAttentionPooling,
    ReduceLayer,
)


class Embedder(nn.Module):
    def __init__(
        self,
        args: Arguments,
        text_field,
        bidirectional: bool = True,
        layer_norm: bool = False,
        highway_bias: float = 0.0,
        pooling: str = "average",
        embedding_dropout: float = 0.1,
        rescale: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """Constructs an model to compute embeddings."""
        super(Embedder, self).__init__()

        # Save values
        self.args = args
        self.device = device
        pad_index = text_field.pad_index()
        self.pad_index = pad_index
        self.embdrop = nn.Dropout(embedding_dropout)
        self.pooling = pooling

        if self.args.bert:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(args.bert_type)
            print("finish loading bert encoder")
            self.emb_size = self.encoder.config.hidden_size
            self.bidirectional = False
            self.bert_bs = args.bert_batch_size
        else:
            num_embeddings = len(text_field.vocabulary)
            self.num_embeddings = num_embeddings
            if args.small_data:
                self.embedding_size = 300
                print("random initializing for debugging")
                self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
            else:
                print(f'Loading embeddings from "{args.embedding_path}"')
                embedding_matrix = text_field.load_embeddings(args.embedding_path)
                self.embedding_size = embedding_matrix.size(1)
                # Create models/parameters
                self.embedding = nn.Embedding(
                    num_embeddings=self.num_embeddings,
                    embedding_dim=self.embedding_size,
                    padding_idx=self.pad_index,
                )
                self.embedding.weight.data = embedding_matrix
            self.embedding.weight.requires_grad = False

            self.bidirectional = bidirectional
            self.layer_norm = layer_norm
            self.highway_bias = highway_bias
            self.rescale = rescale
            self.emb_size = self.args.hidden_size * (1 + self.bidirectional)

            if self.args.encoder == "sru":
                self.encoder = SRU(
                    input_size=self.embedding_size,
                    hidden_size=self.args.hidden_size,
                    num_layers=self.args.num_layers,
                    dropout=self.args.dropout,
                    bidirectional=self.bidirectional,
                    layer_norm=self.layer_norm,
                    rescale=self.rescale,
                    highway_bias=self.highway_bias,
                )

        # if args.word_norm:
        #     self.wordnorm = nn.InstanceNorm1d(self.embedding_size, affine=True)

        # if args.hidden_norm:
        #     self.hiddennorm = nn.InstanceNorm1d(self.emb_size, affine=True)
        self.output_size = self.emb_size

        if self.pooling == "attention":
            self.poollayer = SelfAttentionPooling(
                input_dim=self.output_size,
                attention_heads=self.args.attention_heads,
                attention_units=[self.args.attention_units],
                input_dropout=self.args.dropout,
            )
        else:
            self.poollayer = ReduceLayer(pool=pooling)

        # Move to device
        self.to(self.device)

    def rnn_encode(
        self,
        data: torch.LongTensor,  # batch_size x seq_len
        return_sequence: bool = False,
    ) -> Tuple[List[Tuple[torch.FloatTensor, torch.FloatTensor]], List[Any]]:
        """
        Aligns document pairs.

        :param data: Sentences represented as LongTensors of word indices.
        :param scope: A list of tuples of row_indices and column_indices indexing into data
        to extract the appropriate sentences for each document pair.
        :param data: A list of data for each document pair.
        :return: A tuple consisting of a list of (cost, alignment) tuples and a list of data.
        """
        # Transpose from batch first to sequence first
        data = data.transpose(0, 1)  # seq_len x batch_size

        # Create mask
        mask = (data != self.pad_index).float()  # seq_len x batch_size

        # Embed
        embedded = self.embdrop(
            self.embedding(data)
        )  # seq_len x batch_size x embedding_size

        if self.args.word_norm:
            embedded = F.layer_norm(embedded, embedded.size()[-2:])
            # embedded = self.wordnorm(embedded)

        # RNN encoder
        h_seq, _ = self.encoder(
            embedded, mask_pad=(1 - mask)
        )  # seq_len x batch_size x 2*hidden_size
        # output_states, c_states = sru(x)      # forward pass
        # output_states is (length, batch size, number of directions * hidden size)
        # c_states is (layers, batch size, number of directions * hidden size)

        if self.args.hidden_norm:
            h_seq = F.layer_norm(
                h_seq.transpose(0, 1), h_seq.transpose(0, 1).size()[-2:]
            ).transpose(0, 1)
            # data = self.hiddennorm(data)

        h_seq = h_seq.transpose(0, 1)  # batch_size x seq_len x 2*hidden_size
        mask = mask.transpose(0, 1)

        # return masked_h, masked_h_seq
        if return_sequence:
            masked_h_seq = h_seq * mask.unsqueeze(
                2
            )  # batch_size x seq_len x 2*hidden_size
            return masked_h_seq  # self.project(masked_h_seq)
        else:
            # Average pooling
            # mask = mask.unsqueeze(2)
            # masked_h = masked_h_seq.sum(dim=1)/mask.sum(dim=1) # batch_size x 2*hidden_size
            output = self.poollayer(h_seq, mask)
            return output  # self.project(masked_h)

    def bert_encode(
        self,
        data: torch.LongTensor,  # batch_size x seq_len
        return_sequence: bool = False,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Uses an RNN and self-attention to encode a batch of sequences of word embeddings.
        :param batch: A FloatTensor of shape `(sequence_length, batch_size, embedding_size)` containing embedded text.
        :param lengths: A LongTensor of shape `(batch_size)` containing the lengths of the sequences, used for masking.
        :return: A FloatTensor of shape `(batch_size, output_size)` containing the encoding for each sequence
        in the batch.
        """

        if attention_mask is None and self.pad_index is not None:
            attention_mask = (data != self.pad_index).float()

        attention_mask = attention_mask.to(self.device)
        outputs = self.encoder(data, attention_mask=attention_mask)
        # if not 'distil' in self.args.bert_type:
        masked_h_seq = outputs[0]
        masked_h = outputs[1]
        # else:
        #     masked_h = outputs[0]
        if return_sequence:
            return outputs[0]  # self.project(outputs[0])
        else:
            return outputs[1]  # self.project(outputs[1])

    def forward(
        self,
        data: torch.LongTensor,  # batch_size x seq_len
        return_sequence: bool = False,
    ) -> Tuple[List[Tuple[torch.FloatTensor, torch.FloatTensor]], List[Any]]:
        if self.args.bert:
            # if len(data) > self.bert_bs:
            encodings = []
            batch_size = self.bert_bs
            for batch_idx in range(len(data) // batch_size + 1):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size
                batch = data[start_idx:end_idx]
                # print(data.shape)
                # print(batch.shape)
                if len(batch) == 0:
                    break
                encoded = self.bert_encode(batch, return_sequence)
                # print(encoded.shape)
                encodings.extend(encoded)
                del encoded
            encodings = torch.stack(encodings)
            # print(encodings.shape)
            return encodings
        else:
            return self.rnn_encode(data, return_sequence)

