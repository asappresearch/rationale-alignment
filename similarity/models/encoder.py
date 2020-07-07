from functools import partial
import math
from typing import Any, Iterator, List, Optional, Tuple

from sinkhorn import batch_sinkhorn, construct_cost_and_marginals
from sru import SRU
import torch
import torch.nn as nn
import torch.nn.functional as F

from rationale_alignment.parsing import Arguments
from rationale_alignment.utils import compute_cost, prod, \
    unpad_tensors
from similarity.models.attention import load_attention_layer


class Embedder(nn.Module):
    def __init__(self,
                 args: Arguments,
                 text_field,
                 bidirectional: bool = True,
                 layer_norm: bool = False,
                 highway_bias: float = 0.0,
                 rescale: bool = True,
                 device: torch.device = torch.device('cpu')):
        """Constructs an model to compute embeddings."""
        super(Embedder, self).__init__()

        # Save values
        self.args = args
        self.device = device
        pad_index = text_field.pad_index()
        self.pad_index = pad_index

        if self.args.bert:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(args.bert_type)
            print('finish loading bert encoder')
            self.output_size = self.encoder.config.hidden_size
            self.bidirectional = False
            self.bert_bs = args.bert_batch_size
        else:
            num_embeddings=len(text_field.vocabulary)
            print(f'Loading embeddings from "{args.embedding_path}"')
            embedding_matrix = text_field.load_embeddings(args.embedding_path)

            self.num_embeddings = num_embeddings
            self.embedding_size = embedding_matrix.size(1)
            self.bidirectional = bidirectional
            self.layer_norm = layer_norm
            self.highway_bias = highway_bias
            self.rescale = rescale
            self.word_to_word = args.word_to_word
            self.output_size = self.args.hidden_size * (1 + self.bidirectional)

            # Create models/parameters
            self.embedding = nn.Embedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_size,
                padding_idx=self.pad_index
            )
            self.embedding.weight.data = embedding_matrix
            self.embedding.weight.requires_grad = False

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

        # Move to device
        self.to(self.device)
        


    def rnn_encode(self,
                data: torch.LongTensor,  # batch_size x seq_len
                ) -> Tuple[List[Tuple[torch.FloatTensor, torch.FloatTensor]],
                           List[Any]]:
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
        embedded = self.embedding(data)  # batch_size x seq_len x embedding_size

        # RNN encoder
        h_seq, _ = self.encoder(embedded, mask_pad=(1-mask))  # seq_len x batch_size x 2*hidden_size
        # output_states, c_states = sru(x)      # forward pass
        # output_states is (length, batch size, number of directions * hidden size)
        # c_states is (layers, batch size, number of directions * hidden size)

        masked_h_seq = h_seq * mask.unsqueeze(dim=2) # seq_len x batch_size x 2*hidden_size

        # Average pooling
        masked_h = masked_h_seq.sum(dim=0)/mask.sum(dim=0).unsqueeze(dim=1) # batch_size x 2*hidden_size
        
        masked_h_seq = masked_h_seq.transpose(0,1)
        # return masked_h, masked_h_seq
        return masked_h, None


    def bert_encode(self,
                data: torch.LongTensor,  # batch_size x seq_len
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Uses an RNN and self-attention to encode a batch of sequences of word embeddings.
        :param batch: A FloatTensor of shape `(sequence_length, batch_size, embedding_size)` containing embedded text.
        :param lengths: A LongTensor of shape `(batch_size)` containing the lengths of the sequences, used for masking.
        :return: A FloatTensor of shape `(batch_size, output_size)` containing the encoding for each sequence
        in the batch.
        """
        # print(data.shape)
        # Create mask for padding
        # max_len = lengths.max().item()
        # attention_mask  = torch.zeros(len(data), max_len, dtype=torch.float)
        # for i in range(len(data)):
        #     attention_mask[i, :lengths[i]] = 1

        if attention_mask is None and self.pad_index is not None:
            attention_mask = (data != self.pad_index).float()

        attention_mask = attention_mask.to(self.device)
        outputs = self.encoder(data, attention_mask=attention_mask)
        if not 'distil' in self.args.bert_type:
            masked_h_seq = outputs[0]
            masked_h = outputs[1]
        else:
            masked_h = outputs[0]
        return masked_h, None

    def forward(self,
                data: torch.LongTensor,  # batch_size x seq_len
                ) -> Tuple[List[Tuple[torch.FloatTensor, torch.FloatTensor]],
                           List[Any]]:
        if self.args.bert:
            if len(data) > self.bert_bs:
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
                    encoded, _ = self.bert_encode(batch)
                    # print(encoded.shape)
                    encodings.extend(encoded)
                    del encoded
                encodings = torch.stack(encodings)
                # print(encodings.shape)
            return encodings, None
        else:
            return self.rnn_encode(data)
