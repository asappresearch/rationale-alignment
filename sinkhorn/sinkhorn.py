import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sinkhorn.utils import bmv, compute_masks, mask_log, pad_tensors, p_log_p


def sinkhorn(C: torch.FloatTensor,
             a: Optional[torch.FloatTensor] = None,
             b: Optional[torch.FloatTensor] = None,
             alpha: Optional[torch.FloatTensor] = None,
             beta: Optional[torch.FloatTensor] = None,
             epsilon: float = 1e-3,
             num_iter_max: int = 100,
             unbalanced_lambda: Optional[float] = None,
             error_threshold: float = 1e-5,
             error_check_frequency: int = 10,
             return_log: bool = False) -> Union[torch.FloatTensor,
                                                Tuple[torch.FloatTensor, Dict[str, Any]]]:
    """
    Solves entropic regularization optimal transport using the Sinkhorn-Knopp algorithm.

    Implementation based on https://pot.readthedocs.io/en/stable/_modules/ot/bregman.html#sinkhorn_stabilized
    and on https://arxiv.org/abs/1803.00567 (Section 4.2).

    Supports batching by adding an extra first dimension.

    Note: A marginal (a or b) of None defaults to uniform.

    :param C: Cost matrix (n x m or num_batches x n x m). Note: The device and dtype of C are used for all other variables.
    :param a: Row marginals (n or num_batches x n).
    :param b: Column marginals (m or num_batches x m).
    :param alpha: Initial value for alpha log scaling stability (n or num_batches x n).
    :param beta: Initial value for beta log scaling stability (m or num_batches x m).
    :param epsilon: Weighting factor of entropy regularization (higher = more entropy, lower = less entropy).
    :param num_iter_max: Maximum number of iterations to perform.
    :param unbalanced_lambda: The weighting factor which controls marginal divergence for unbalanced OT.
    :param error_threshold: Marginal error at which to stop sinkhorn iterations.
    :param error_check_frequency: Frequency with with to check the error.
    :param return_log: Whether to also return a dictionary with logging information.
    :return: The optimal alignment matrix (n x m or num_batches x n x m) according to the cost and marginals
    and optionally a dictionary of logging information.
    """
    # Setup
    dtype = C.dtype
    device = C.device

    # Get shape and whether it's batched
    if len(C.shape) == 3:
        batched = True
        nb, n, m = C.shape
        n_shape = (nb, n)
        m_shape = (nb, m)
    elif len(C.shape) == 2:
        batched = False
        n, m = C.shape
        n_shape = (n,)
        m_shape = (m,)
    else:
        raise ValueError(f'C must have 2 or 3 dimensions, got {len(C.shape)}')

    # Use uniform distribution if marginals not provided
    if a is None:
        a = torch.ones(n_shape, dtype=dtype, device=device) / n
    a = a.to(C)

    if b is None:
        b = torch.ones(m_shape, dtype=dtype, device=device) / m
    b = b.to(C)

    # Construct mask (1s for content, 0s for padding)
    mask, mask_n, mask_m = compute_masks(C=C, a=a, b=b)

    # Initialize alpha and beta if not provided (scaling values in log domain to help stability)
    if alpha is None:
        alpha = torch.zeros(n_shape, dtype=dtype, device=device)
    alpha.to(C)

    if beta is None:
        beta = torch.zeros(m_shape, dtype=dtype, device=device)
    beta.to(C)

    # Set u and v
    if batched:
        u_init, v_init = mask_n, mask_m
    else:
        u_init = torch.ones(n_shape, dtype=dtype, device=device)
        v_init = torch.ones(m_shape, dtype=dtype, device=device)
    u, v = u_init, v_init

    # Check shapes
    assert a.shape == alpha.shape == u.shape == mask_n.shape == n_shape
    assert b.shape == beta.shape == v.shape == mask_m.shape == m_shape
    assert C.shape == mask.shape

    # Define functions to compute K and P
    def compute_log_K(alpha: torch.FloatTensor,
                      beta: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes log(K) = -C / epsilon.

        :param alpha: Alpha log scaling factor.
        :param beta: Beta log scaling factor.
        :return: log(K) = -C / epsilon with stability and masking.
        """
        return mask * (-C + alpha.unsqueeze(dim=-1) + beta.unsqueeze(dim=-2)) / epsilon

    def compute_K(alpha: torch.FloatTensor,
                  beta: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes K = exp(-C / epsilon).

        :param alpha: Alpha log scaling factor.
        :param beta: Beta log scaling factor.
        :return: K = exp(-C / epsilon) with stability and masking.
        """
        return torch.exp(compute_log_K(alpha, beta))

    def compute_P(alpha: torch.FloatTensor,
                  beta: torch.FloatTensor,
                  u: torch.FloatTensor,
                  v: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes transport matrix P = diag(u) * K * diag(v).

        :param alpha: Alpha log scaling factor.
        :param beta: Beta log scaling factor.
        :param u: u vector.
        :param v: v vector.
        :return: P = diag(u) * K * diag(v) with stability and masking.
        """
        return mask * torch.exp(mask_log(u, mask_n).unsqueeze(dim=-1) +
                                compute_log_K(alpha, beta) +
                                mask_log(v, mask_m).unsqueeze(dim=-2))

    # Initialize K
    K = compute_K(alpha, beta)  # nb x n x m
    K_t = K.transpose(-2, -1)  # nb x m x n

    # Set lambda for unbalanced OT
    if unbalanced_lambda is not None:
        unbalanced_lambda = unbalanced_lambda / (unbalanced_lambda + epsilon)

    # Set matrix-vector product function
    mv = bmv if batched else torch.mv

    # Set error and num_iter
    error = float('inf')
    num_iter = 0

    # Sinkhorn iterations
    while error > error_threshold and num_iter < num_iter_max:
        # Save previous u and v in case of numerical errors
        u_prev, v_prev = u, v

        # Sinkhorn update
        u = (a / (mv(K, v) + 1e-16))
        if unbalanced_lambda is not None:
            u **= unbalanced_lambda

        v = (b / (mv(K_t, u) + 1e-16))
        if unbalanced_lambda is not None:
            v **= unbalanced_lambda

        # Check if we've broken machine precision and if so, return previous result
        unstable = torch.isnan(u).any(dim=-1, keepdim=True) | torch.isnan(v).any(dim=-1, keepdim=True) | \
                   torch.isinf(u).any(dim=-1, keepdim=True) | torch.isinf(v).any(dim=-1, keepdim=True)
        if unstable.any():
            print(f'Warning: Numerical errors at iteration {num_iter}')
            # print(f'shape: {m},{n}')
            # print(f'epsilon: {epsilon}')
            # print(f'error threshold: {error_threshold}')
            # print(C.max())
            # print(C.min())

            if batched:
                u = torch.where(unstable.repeat(1, u.size(-1)), u_prev, u)
                v = torch.where(unstable.repeat(1, v.size(-1)), v_prev, v)
            else:
                u, v = u_prev, v_prev

            if unstable.all():
                break

        # Remove numerical problems in u and v by moving them to K
        alpha += epsilon * mask_log(u, mask_n)
        beta += epsilon * mask_log(v, mask_m)
        K = compute_K(alpha, beta)
        K_t = K.transpose(-2, -1)
        u, v = u_init, v_init

        # Check error
        if num_iter % error_check_frequency == 0:
            P = compute_P(alpha, beta, u, v)
            errors = torch.norm(P.sum(dim=-1) - a, dim=-1) ** 2
            error = errors.max()

        # Update num_iter
        num_iter += 1

    # Compute optimal transport matrix
    P = compute_P(alpha, beta, u, v)

    if return_log:
        log = {
            'alpha': alpha,
            'beta': beta,
            'error': error,
            'num_iter': num_iter
        }
        return P, log

    return P


def sinkhorn_epsilon_scaling(C: torch.FloatTensor,
                             a: Optional[torch.FloatTensor] = None,
                             b: Optional[torch.FloatTensor] = None,
                             alpha: Optional[torch.FloatTensor] = None,
                             beta: Optional[torch.FloatTensor] = None,
                             epsilon: float = 1e-3,
                             num_iter_max: int = 100,
                             unbalanced_lambda: Optional[float] = None,
                             error_threshold: float = 1e-5,
                             error_check_frequency: int = 10,
                             num_scaling_min: int = 10,
                             num_scaling_max: int = 15,
                             epsilon_0: float = 0.1) -> torch.FloatTensor:
    """
    Solves entropic regularization optimal transport using the Sinkhorn-Knopp algorithm with epsilon scaling.

    Implementation based on https://pot.readthedocs.io/en/stable/_modules/ot/bregman.html#sinkhorn_epsilon_scaling

    Supports batching by adding an extra first dimension.

    Note: A marginal (a or b) of None defaults to uniform.

    :param C: Cost matrix (n x m or num_batches x n x m). Note: The device and dtype of C are used for all other variables.
    :param a: Row marginals (n or num_batches x n).
    :param b: Column marginals (m or num_batches x m).
    :param alpha: Initial value for alpha log scaling stability (n or num_batches x n).
    :param beta: Initial value for beta log scaling stability (m or num_batches x m).
    :param epsilon: Weighting factor of entropy regularization (higher = more entropy, lower = less entropy).
    :param num_iter_max: Maximum number of iterations to perform.
    :param unbalanced_lambda: The weighting factor which controls marginal divergence for unbalanced OT.
    :param error_threshold: Marginal error at which to stop sinkhorn iterations.
    :param error_check_frequency: Frequency with with to check the error.
    :param num_scaling_min: The minimum number of epsilon scaling steps (i.e. applications of sinkhorn).
    :param num_scaling_max: The maximum number of epsilon scaling steps (i.e. applications of sinkhorn).
    :param epsilon_0: The initial epsilon.
    :return: The optimal alignment matrix (n x m or nb x n x m) according to the cost and marginals.
    """
    def get_epsilon(step: int):
        """Exponentially decreasing from epsilon_0 to epsilon."""
        return (epsilon_0 - epsilon) * math.exp(-step) + epsilon

    for step in range(num_scaling_max):
        P, log = sinkhorn(
            C=C,
            a=a,
            b=b,
            alpha=alpha,
            beta=beta,
            epsilon=get_epsilon(step),
            num_iter_max=num_iter_max,
            unbalanced_lambda=unbalanced_lambda,
            error_threshold=error_threshold,
            error_check_frequency=error_check_frequency,
            return_log=True
        )

        if step > num_scaling_min and log['error'] < error_threshold:
            break

        alpha, beta = log['alpha'], log['beta']

    return P


def batch_sinkhorn(C_list: List[torch.FloatTensor],
                   a_list: Optional[List[torch.FloatTensor]] = None,
                   b_list: Optional[List[torch.FloatTensor]] = None,
                   alpha_list: Optional[List[torch.FloatTensor]] = None,
                   beta_list: Optional[List[torch.FloatTensor]] = None,
                   epsilon_scaling: bool = True,
                   return_tensor: bool = False,
                   **kwargs) -> List[torch.FloatTensor]:
    """
    Performs batched sinkhorn or sinkhorn_epsilon_scaling.

    :param C_list: A list of cost matrices (n x m). Note: The device and dtype of C_list[0] are used for all other variables.
    :param a_list: A list of row marginals (n).
    :param b_list: A list of column marginals (m).
    :param alpha_list: A list of initial values for alpha log scaling stability (n).
    :param beta_list: A list of initial values for beta log scaling stability (m).
    :param epsilon_scaling: Whether to use epsilon scaling.
    :param kwargs: Remaining keyword arguments for sinkhorn or sinkhorn_epsilon_scaling.
    :return: A list of optimal alignment matrices (n x m).
    """
    # Get device and dtype
    dtype = C_list[0].dtype
    device = C_list[0].device

    # Get sizes
    n_list, m_list = zip(*(C.shape for C in C_list))

    # Initialize marginals with uniform if not provided
    if a_list is None:
        a_list = [torch.ones(n, dtype=dtype, device=device) / n for n in n_list]

    if b_list is None:
        b_list = [torch.ones(m, dtype=dtype, device=device) / m for m in m_list]

    # Initialize alpha/beta with zero if not provided
    if alpha_list is None:
        alpha_list = [torch.zeros(n, dtype=dtype, device=device) for n in n_list]

    if beta_list is None:
        beta_list = [torch.zeros(m, dtype=dtype, device=device) for m in m_list]

    # Batch everything by padding with zeros
    C_batch = pad_tensors(C_list)
    a_batch = pad_tensors(a_list)
    b_batch = pad_tensors(b_list)
    alpha_batch = pad_tensors(alpha_list)
    beta_batch = pad_tensors(beta_list)

    # Run batched sinkhorn
    sinkhorn_algo = sinkhorn_epsilon_scaling if epsilon_scaling else sinkhorn
    P_batch = sinkhorn_algo(
        C=C_batch,
        a=a_batch,
        b=b_batch,
        alpha=alpha_batch,
        beta=beta_batch,
        **kwargs
    )

    if return_tensor:
        return P_batch
    else:
        # Extract alignment matrices and throw away padding
        P_list = [P_batch[i, :n, :m] for i, (n, m) in enumerate(zip(n_list, m_list))]
        return P_list



def compute_entropy(P: torch.FloatTensor) -> torch.float:
    """
    Computes the entropy of P: H(P) = sum_{ij} P_ij * log(P_ij).

    :param P: A probability matrix.
    :return: The entropy of P.
    """
    return -torch.sum(p_log_p(P))


def compute_alignment_cost(C: torch.FloatTensor,
                           P: torch.FloatTensor,
                           include_entropy: bool = False,
                           epsilon: Optional[float] = None) -> torch.float:
    """
    Computes the cost of alignment P according to cost matrix C: <C, P>.

    :param C: The cost matrix (n x m).
    :param P: The alignment matrix (n x m).
    :param include_entropy: Whether to include the entropy regularization in the cost.
    :param epsilon: Weighting factor of entropy regularization (higher = more entropy, lower = less entropy).
    :return: The cost of the alignment P according to cost matrix C (i.e. <C, P>).
    """
    assert include_entropy == (epsilon is not None)

    cost = torch.sum(C * P)

    if include_entropy:
        cost -= epsilon * compute_entropy(P)

    return cost
