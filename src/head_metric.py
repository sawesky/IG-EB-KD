from typing import Callable, Optional, Tuple

import torch

# append a constant 1 to each feature vector for the bias coordinate, so it becomes [B, d + 1]
def augment_features(features: torch.Tensor) -> torch.Tensor:

    ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
    return torch.cat([features, ones], dim=1)

# apply the minibatch final-layer Fisher pullback to a head-shaped vector F_bar[V] = (1/B) sum_i G_i (V h_i) h_i^T
def fisher_action(
    V: torch.Tensor,
    h_aug: torch.Tensor,
    probs: torch.Tensor,
) -> torch.Tensor:

    batch_size, dim = h_aug.shape

    # r_i = V h_i, vectorized over the minibatch: [B, K]
    R = h_aug @ V.transpose(0, 1)

    # G_i r_i = p_i <hadamard> r_i - p_i (p_i^T r_i), without constructing G_i
    p_dot_r = (probs * R).sum(dim=1, keepdim=True)
    U = probs * R - probs * p_dot_r

    # (1/B) sum_i u_i h_i^T: [K, D]
    return U.transpose(0, 1) @ h_aug / batch_size

# apply A[V] = V + rho * F_bar[V]
def metric_action(
    V: torch.Tensor,
    h_aug: torch.Tensor,
    probs: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    return V + rho * fisher_action(V, h_aug, probs)

# solve Ax = b with conjugate gradient for an SPD matrix-free operator, unknown may have any shape (vector/matrix)
def conjugate_gradient(
    operator: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    x0: Optional[torch.Tensor] = None,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, int, float]:
    
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - operator(x)
    p = r.clone()

    rs_old = torch.sum(r * r)
    b_norm = torch.linalg.vector_norm(b)
    scale = torch.clamp(b_norm, min=torch.finfo(b.dtype).eps)
    rel_residual = torch.sqrt(rs_old) / scale

    if rel_residual.item() <= tol:
        return x, 0, rel_residual.item()

    for iteration in range(1, max_iter + 1):
        Ap = operator(p)
        pAp = torch.sum(p * Ap)
        if pAp.item() <= 0.0:
            raise RuntimeError(
                "CG encountered a non-positive curvature direction; "
                "the supplied operator may not be SPD."
            )

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.sum(r * r)
        rel_residual = torch.sqrt(rs_new) / scale
        if rel_residual.item() <= tol:
            return x, iteration, rel_residual.item()

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, rel_residual.item()

# pack [grad_W | grad_b] in the same row-wise augmented shape as the metric
def pack_linear_head_gradient(head: torch.nn.Linear) -> torch.Tensor:
    return torch.cat(
        [head.weight.grad.detach(), head.bias.grad.detach().unsqueeze(1)],
        dim=1,
    )

# apply [W | b] <- [W | b] - lr * direction
@torch.no_grad()
def apply_linear_head_direction(
    head: torch.nn.Linear,
    direction: torch.Tensor,
    lr: float,
) -> None:
    
    head.weight.add_(direction[:, :-1], alpha=-lr)
    head.bias.add_(direction[:, -1], alpha=-lr)

# Solve (I + rho F_bar) d = gradient for a linear classification head, no gradients are propagated through the curvature solve
def solve_metric_direction(
    gradient: torch.Tensor,
    features: torch.Tensor,
    probs: torch.Tensor,
    rho: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, int, float]:
    
    gradient = gradient.detach()
    h_aug = augment_features(features.detach())
    probs = probs.detach()

    if rho == 0.0:
        return gradient.clone(), 0, 0.0

    def operator(V: torch.Tensor) -> torch.Tensor:
        return metric_action(V, h_aug, probs, rho)

    return conjugate_gradient(
        operator,
        gradient,
        tol=tol,
        max_iter=max_iter,
    )
