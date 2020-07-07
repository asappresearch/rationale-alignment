from typing import Optional, Tuple

import torch

from sinkhorn.utils import compute_masks


def construct_cost_and_marginals(C: torch.FloatTensor,
                                 one_to_k: Optional[int] = None,
                                 max_num_aligned: Optional[int] = None,
                                 optional_alignment: bool = False,
                                 split_dummy: bool = False,
                                 order_lambda: Optional[float] = None,
                                 ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Given a cost matrix, adds dummy nodes (if necessary) and returns the new cost matrix and the row and column marginals.
    
    Dummy nodes always have cost 0.

    :param C: FloatTensor (n x m) with costs.
    :param one_to_k: The k for one-to-k alignment, where each node on the smaller side aligns to k nodes on the larger side.
    If k is -1, uses the maximum possible k, which is k = floor(max(n, m) / min(n, m)).
    :param max_num_aligned: Whether to constrain the total number of alignments to this number.
    :param optional_alignment: Whether to allow every node to align to a dummy node. Requires a cost that can be + or -.
    :param split_dummy: Whether to split multi-marginal dummy nodes into multiple nodes with marginal 1 each.
    :param order_lambda: Weight for an added diagonal order preserving cost (0 = no weight, 1 = all weight).
    :return: A tuple containing the cost matrix, the row marginals, and the column marginals.
    """
    # Checks
    assert len(C.shape) == 2
    assert one_to_k is None or max_num_aligned is None
    if max_num_aligned is not None:
        assert not optional_alignment

    # Setup
    n, m = C.shape
    device = C.device

    # Ensure n >= m for all the following for convenience, then return to original order at the end
    swap = n < m
    if swap:
        n, m = m, n

    # Default marginals
    a = [1] * n
    b = [1] * m

    # If max_num_aligned is more than a one-to-one alignment, just do a one-to-one alignment
    if max_num_aligned is not None and max_num_aligned >= m:
        max_num_aligned = None
        one_to_k = 1

    # If optional alignment, set one-to-one because it'll happen anyway
    if optional_alignment and one_to_k is None:
        one_to_k = 1

    # Dummy nodes
    if max_num_aligned is not None:
        assert max_num_aligned > 0

        if split_dummy:
            a = b = [1] * (n + m - max_num_aligned)
        else:
            a = [1] * n + [m - max_num_aligned]
            b = [1] * m + [n - max_num_aligned]
    elif one_to_k is not None:
        assert one_to_k == -1 or one_to_k > 0

        # Add enough dummy nodes to absorb the non-divisibility of n and m
        max_k = n // m
        k = max_k if one_to_k == -1 else min(one_to_k, max_k)
        rem = n - k * m

        if optional_alignment:
            if split_dummy:
                a = [1] * n + [k] * m
                b = [k] * m + [1] * n
            else:
                a = [1] * n + [k * m]
                b = [k] * m + [n]
        elif rem != 0:
            if split_dummy:
                a = [1] * n
                b = [k] * m + [1] * rem
            else:
                a = [1] * n
                b = [k] * m + [rem]

    # Return to original ordering
    if swap:
        n, m = m, n
        a, b = b, a

    # Add zero padding for dummy node cost if there are dummy nodes
    num_rows, num_cols = len(a), len(b)
    if (num_rows, num_cols) != (n, m):
        padded_cost = torch.zeros(num_rows, num_cols, device=device)
        padded_cost[:n, :m] = C
        C = padded_cost

    # Tensorize marginals and normalize to one
    a = torch.FloatTensor(a).to(device) / sum(a)
    b = torch.FloatTensor(b).to(device) / sum(b)

    # Add diagonal order-preserving cost
    if order_lambda is not None:
        C = add_order_cost(C=C, a=a, b=b, order_lambda=order_lambda)

    return C, a, b


def add_order_cost(C: torch.FloatTensor,
                   a: torch.FloatTensor,
                   b: torch.FloatTensor,
                   order_lambda: float) -> torch.FloatTensor:
    """
    Adds diagonal order preserving cost to a cost matrix.

    :param C: FloatTensor (n x m or num_batches x n x m) with costs.
    Note: The device and dtype of C are used for all other variables.
    :param a: Row marginals (num_batches x n).
    :param b: Column marginals (num_batches x m).
    :param order_lambda: Weight for diagonal order preserving cost (0 = no weight, 1 = all weight).
    :return: The cost matrix with the order preserving cost added in.
    """
    # Checks
    assert len(C.shape) in [2, 3]
    batched = len(C.shape) == 3
    assert len(a.shape) == len(b.shape) == (2 if batched else 1)
    assert 0.0 <= order_lambda <= 1.0

    # Return C if not adding order cost
    if order_lambda == 0.0:
        return C

    # Setup
    dtype = C.dtype
    device = C.device

    # Compute order preserving cost
    I = torch.arange(C.size(-2), dtype=dtype, device=device).unsqueeze(dim=1)
    J = torch.arange(C.size(-1), dtype=dtype, device=device).unsqueeze(dim=0)

    # Compute masks
    mask, mask_n, mask_m = compute_masks(C=C, a=a, b=b)

    # Compute N and M
    N = mask_n.sum(dim=-1, keepdim=True)
    M = mask_m.sum(dim=-1, keepdim=True)

    if batched:
        I = I.unsqueeze(dim=0).repeat(C.size(0), 1, C.size(2))
        J = J.unsqueeze(dim=0).repeat(C.size(0), C.size(1), 1)

        N = N.unsqueeze(dim=-1)
        M = M.unsqueeze(dim=-1)
    else:
        I = I.repeat(1, C.size(1))
        J = J.repeat(C.size(0), 1)

    D = torch.abs(I / N - J / M)

    # Match average magnitudes so costs are on the same scale
    mask_sum = mask.sum(dim=-1).sum(dim=-1)
    C_magnitude = (mask * C.abs()).sum(dim=-1).sum(dim=-1) / mask_sum
    D_magnitude = (mask * D.abs()).sum(dim=-1).sum(dim=-1) / mask_sum
    D_magnitude[D_magnitude == 0] = 1  # prevent divide by 0
    D *= (C_magnitude / D_magnitude).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # Add order preserving cost
    C = (1 - order_lambda) * C + order_lambda * D
    C *= mask

    return C
