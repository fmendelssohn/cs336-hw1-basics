from collections.abc import Callable, Iterable 
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    """
    Implements SGD with learning-rate divided by sqrt(t+1).
    """
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        # closure can be used to recompute the loss before the optimizer step
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with parameter p
                t = state.get("t", 0)  # Get iteration number or initial value from state
                grad = p.grad.data  # Get the gradient of loss w.r.t. p
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place
                state["t"] = t+1  

        return loss


# uv run pytest -k test_adamw
class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params: Iterable[torch.nn.parameter.Parameter], 
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                alpha = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                t = state.get('t', 1)  # Iteration number 

                prev_m_t = state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad))

                # Update first and second moment estimates
                m_t = beta1 * prev_m_t + (1-beta1) * grad  
                v_t = beta2 * prev_v_t + (1-beta2) * torch.square(grad)

                # Compute adjusted learning-rate for iteration; note not persistent
                alpha_t = alpha * math.sqrt(1 - beta2**t) / (1 - beta1**t)  

                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)  # Update parameters
                p.data.mul_(1 - alpha * weight_decay)  # Apply weight-decay 

                # Record
                state['m'] = m_t
                state['v'] = v_t
                state['t'] = t+1

        return loss


# uv run pytest -k test_get_lr_cosine_schedule
def get_cosine_lr(
    t: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int,
) -> float:
    """
    Behavior:
        linear increasing up for warmup_iters;
        cosine decreasing down up to cosine_cycle_iters; 
        constant at min_learning_rate afterwards.
    """
    if t < warmup_iters:
        return t/warmup_iters * max_learning_rate
    elif warmup_iters <= t <= cosine_cycle_iters:
        decay_ratio = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coef = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_learning_rate + coef * (max_learning_rate - min_learning_rate)
    else:  # t > cosine_cycle_iters
        return min_learning_rate
