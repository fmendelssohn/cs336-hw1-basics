from typing import Iterable
from pytest import param
import torch
import torch.nn as nn
from torch import Tensor, optim
import einx
import os
import typing
import numpy as np
from jaxtyping import Float, Bool, Int


def silu(x: Tensor):
    return x * torch.sigmoid(x)
    

# uv run pytest -k test_softmax_matches_pytorch
def softmax(x: Tensor, dim: int = -1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def log_softmax(x: Tensor, dim: int = -1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    z = x - x_max
    # torch.logsumexp
    return z - torch.log(torch.sum(torch.exp(z), dim=dim, keepdim=True))
    

# uv run pytest -k test_cross_entropy
def cross_entropy(logits: Float[Tensor, " batch_size seq_len vocab_size"], targets: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, ""]:
    """
    Args:
        logits (Float[Tensor, "batch_size vocab_size"]): logits[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns: Average cross-entropy loss across examples.
    """
    scores: Float[Tensor, " batch_size seq_len vocab_size"]  = -log_softmax(logits)
    # Append new dimension to targets to match vocab_size dim;
    # Index into scores with targets values 
    values: Float[Tensor, " batch_size seq_len"] = torch.gather(scores, dim=-1, index=targets.unsqueeze(-1))
    return torch.mean(values)  # torch(float)


# uv run pytest -k test_gradient_clipping
def clip_gradients(parameters: Iterable[torch.nn.parameter.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Gradient clipping on the L2-norm of the gradients over all parameters. 
    """
    # NOTE: p.grad.data bypasses autograd and can cause bugs
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum((g**2).sum() for g in grads))

    clip_coef = min(1, max_l2_norm/(norm + eps))
    for g in grads:  # Modifies in-place
        g.mul_(clip_coef)
    return 


# uv run pytest -k test_checkpointing
def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    state_dict = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
    )
    torch.save(state_dict, out)
    return


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> int:
    state_dict = torch.load(src)
    assert {'model', 'optimizer', 'iteration'}.issubset(state_dict.keys())
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return state_dict['iteration']
