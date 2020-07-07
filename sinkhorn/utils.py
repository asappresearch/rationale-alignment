from typing import List, Optional, Tuple

import torch


def bmv(m: torch.FloatTensor, v: torch.FloatTensor) -> torch.FloatTensor:
    """
    Performs a batched matrix-vector product.

    :param m: A 3-dimensional FloatTensor (num_batches x n1 x n2).
    :param v: A 2-dimensional FloatTensor (num_batches x n2).
    :return: Batched matrix-vector product mv (num_batches x n1).
    """
    assert len(m.shape) == 3
    assert len(v.shape) == 2
    return torch.bmm(m, v.unsqueeze(dim=2)).squeeze(dim=2)


def compute_masks(C: torch.FloatTensor,
                  a: torch.FloatTensor,
                  b: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Computes masks for C, a, and b based on the zero entries in a and b.

    Masks have 1s for content and 0s for padding.

    :param C: Cost matrix (n x m or num_batches x n x m).
    Note: The device and dtype of C are used for all other variables.
    :param a: Row marginals (n or num_batches x n).
    :param b: Column marginals (m or num_batches x m).
    :return: A tuple containing a mask for C, a mask for a, and a mask for b.
    """
    mask_n = (a != 0)
    mask_m = (b != 0)
    mask = mask_n.unsqueeze(dim=-1) & mask_m.unsqueeze(dim=-2)
    mask, mask_n, mask_m = mask.to(C), mask_n.to(C), mask_m.to(C)

    return mask, mask_n, mask_m


def pad_tensors(tensor_list: List[torch.FloatTensor],
                padding: float = 0.0) -> torch.FloatTensor:
    """
    Pads and stacks a list of tensors, each with the same number of dimensions.

    :param tensor_list: A list of FloatTensors to pad.
    Note: The device and dtype of the first tensor are used for all other variables.
    :param padding: Padding value to use.
    :return: A FloatTensor containing the padded and stacked tensors in tensor_list.
    """
    # Determine maximum size along each dimension
    shape_list = [tensor.shape for tensor in tensor_list]
    shape_max = torch.LongTensor(shape_list).max(dim=0)[0]

    # Create padding with shape (num_batches, *shape_max)
    tensor_batch = padding * torch.ones(len(tensor_list), *shape_max,
                                        dtype=tensor_list[0].dtype, device=tensor_list[0].device)

    # Put content of tensors into the batch tensor
    for i, (tensor, shape) in enumerate(zip(tensor_list, shape_list)):
        tensor_slice = [i, *[slice(size) for size in shape]]
        tensor_batch[tensor_slice] = tensor

    return tensor_batch


def mask_log(x: torch.FloatTensor, mask: Optional[torch.Tensor] = None) -> torch.FloatTensor:
    """
    Takes the logarithm such that the log of masked entries is zero.

    :param x: FloatTensor whose log will be computed.
    :param mask: Tensor with 1s for content and 0s for padding.
    Entries in x corresponding to 0s will have a log of 0.
    :return: log(x) such that entries where the mask is 0 have a log of 0.
    """
    if mask is not None:
        # Set masked entries of x equal to 1 (in a differentiable way) so log(1) = 0
        mask = mask.float()
        x = x * mask + (1 - mask)

    return torch.log(x)


def p_log_p(p: torch.FloatTensor) -> torch.FloatTensor:
    """
    Computes p * log(p) so that 0 * log(0) = 0.

    :param p: A FloatTensor of probabilities between 0 and 1.
    :return: The elementwise computation p * log(p).
    """
    return p * mask_log(p, mask=p != 0)
