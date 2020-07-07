from typing import Optional, Iterable

import torch
from torch import nn



class SelfAttentionPooling(nn.Module):
    """Self attention pooling."""
    def __init__(self,
                 input_dim: int,
                 attention_heads: int = 16,
                 attention_units: Optional[Iterable[int]] = None,
                 output_activation: Optional[torch.nn.Module] = None,
                 hidden_activation: Optional[torch.nn.Module] = None,
                 input_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 ):
        """Initialize a self attention pooling layer

        Parameters
        ----------
        input_dim : int
            The input data dim
        attention_heads: int
            the number of attn heads
        attention_units: Iterable[int]
            the list of hidden dimensions of the MLP computing the attn
        input_dropout: float
            dropout applied to the data argument of the forward method.
        attention_dropout: float
            dropout applied to the attention output before applying it
            to the input for reduction. decouples the attn dropout
            from the input dropout
        """
        super().__init__()
        # creating the MLP
        dimensions = [input_dim, *attention_units, attention_heads]
        self.input_dim = input_dim
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0. else nn.Identity()
        layers = []
        for l in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[l], dimensions[l+1], bias=False))
            layers.append(nn.Tanh() if hidden_activation is None else hidden_activation)
        layers.append(nn.Linear(dimensions[-2], dimensions[-1], bias=False))
        if attention_dropout > 0.:
            layers.append(nn.Dropout(attention_dropout))
        self.mlp = nn.Sequential(*layers)
        self.output_activation = nn.Softmax(dim=1) \
            if output_activation is None else output_activation

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input padding_mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # input_tensor is 3D float tensor, batchsize x num_encs x dim
        batch_size, num_encs, dim = data.shape
        # apply input droput
        data = self.in_drop(data)
        # apply projection and reshape to batchsize x num_encs x num_heads
        attention_logits = self.mlp(data.reshape(-1, dim)).reshape(batch_size, num_encs, -1)
        # apply padding_mask. dimension stays batchsize x num_encs x num_heads
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(2).float()
            attention_logits = attention_logits * padding_mask + (1. - padding_mask) * -1e20
        # apply softmax. dimension stays batchsize x num_encs x num_heads
        attention = self.output_activation(attention_logits)
        # attend. attention is batchsize x num_encs x num_heads. data is batchsize x num_encs x dim
        # resulting dim is batchsize x num_heads x dim
        attended = torch.bmm(attention.transpose(1, 2), data) 
        # average over attention heads and return. dimension is batchsize x dim
        return attended.mean(dim=1)


class ReduceLayer(nn.Module):
    """Implement an sigmoid module.

    Can be used to form a classifier out of any encoder.
    Note: by default takes the log_softmax so that it can be fed to
    the NLLLoss module. You can disable this behavior through the
    `take_log` argument.

    """
    def __init__(self, 
                pool: str='average', 
                reduce_dim: int = 1, 
                padding_idx: Optional[int] = 0) -> None:
        """Initialize the SoftmaxLayer.

        Parameters
        ----------
        """
        super().__init__()
        # output of nn.embedding: B X S X E
        # input and output of RNN: S X B X H
        # Padding mask: B X S
        self.reduce_dim = reduce_dim  # Most of time, output is B x S x E, with seqlength on dimension 1
        self.pool = pool
        # self.padding_idx = padding_idx


    def forward(self,
                data: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S]

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor of shape [B x H]

        """
        output = data
        # print('input')
        # print(output.shape)
        if padding_mask is None:
            padding_mask = torch.ones(*output.shape[:2]).to(output)
            
        # print('mask')
        # print(padding_mask.shape)

        # cast(torch.Tensor, padding_mask)
        if self.pool == 'average':
            # print(padding_mask.shape)
            # print(data.shape)
            padding_mask = padding_mask.unsqueeze(2)
            output = (output * padding_mask).sum(dim=self.reduce_dim) #BXE
            output = output / padding_mask.sum(dim=self.reduce_dim)
        elif self.pool == 'sum':
            output = (output * padding_mask.unsqueeze(2)).sum(dim=self.reduce_dim)
        elif self.pool == 'last':
            lengths = padding_mask.long().sum(dim=self.reduce_dim)
            output = output[torch.arange(output.size(0)).long(), lengths - 1, :]
        elif self.pool == 'first':
            output = output[torch.arange(output.size(0)).long(), 0, :]
        elif self.pool == 'sqrt_reduction':
            '''original implementation can be found here 
            https://github.asapp.dev/aganatra/nlp/blob/master/src/agnlp/utils/sqrt_n_reduction.py'''
            padding_mask = padding_mask.unsqueeze(2)
            output = (output * padding_mask).sum(dim=self.reduce_dim) #BXE
            output = output/sqrt(padding_mask.sum(dim=self.reduce_dim).float())
        # elif self.pool == 'decay':
        #     xxxx
        else:
            pool = self.pool
            print(pool)
            raise ValueError(f"Invalid pool type: {pool}")

        return output

