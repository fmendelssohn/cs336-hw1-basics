"""
Reference implementations:
https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336-basics/cs336_basics/model.py
https://github.com/ZitongYang/cs336-assignment1-basics/tree/master/cs336_basics
"""
import torch
import torch.nn as nn
from typing import Iterable
from torch import Tensor
import einx
import numpy as np
from jaxtyping import Float, Bool, Int
from cs336_basics.utils.nn_utils import silu, softmax
from einx import rearrange

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# uv run pytest -k test_linear
class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype

        # Initialize weights 
        sd = np.sqrt(2. / (self.d_in + self.d_out))
        w_init = torch.empty(self.d_out, self.d_in, device=self.device, dtype=self.dtype)
        nn.init.trunc_normal_(w_init, mean=0, std=sd, a=-3*sd, b=3*sd)
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(w_init, requires_grad=True)
        return
    
    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        """
        """
        if x.device != self.device or x.dtype != self.dtype:
            x.to(device=self.device, dtype=self.dtype)
        # No bias term
        return einx.dot("... d_in, d_out d_in -> ... d_out", x, self.weight)


# uv run pytest -k test_embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embedding vectors 
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embedding matrix
        sd = 1.
        embeddings_init = nn.init.trunc_normal_(
            torch.empty(self.vocab_size, self.d_model), 
            mean=0, std=sd, a=-3*sd, b=3*sd)
        self.weight = nn.Parameter(embeddings_init, requires_grad=True)
        return

    def forward(self, token_ids: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]


# uv run pytest -k test_rmsnorm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """
            d_model: Hidden dimension of the model.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device))
        self.eps = eps
        return
    
    def forward(self, x: Float[Tensor, " ... seq_len d_model"]) -> Float[Tensor, " ... seq_len d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Upcast to prevent overflow

        # Perform RMSNorm
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x * rms).to(in_dtype)


# uv run pytest -k test_swiglu
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            # Approximately 8/3 * d_model, round to nearest multiple of 64
            self.d_ff = int((8/3 * d_model + 32) // 64) * 64  
        else:
            self.d_ff = d_ff
        
        # Linear weights
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        return

    def forward(self, x: Float[Tensor, " ... d_model"]):
        return self.w2(silu(self.w1(x)) * self.w3(x))


# uv run pytest -k test_rope
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, dim: int, context_length: int, device: torch.device | None = None):
        super().__init__()
        self.register_buffer(
            "_freq_cls_cache", 
            RotaryPositionalEmbedding._init_cache(context_length, dim, theta), 
            persistent=False,
        )
        return 

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0
        d = torch.arange(0, dim, 2) / dim
        f = torch.pow(theta, -d)
        t = torch.arange(context_length)
        freqs = einx.dot("t, f -> t f", t, f)
        return torch.stack((torch.cos(freqs), torch.sin(freqs)))

    def forward(self, x: Float[Tensor, "... seq d"], pos_ids: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq d"]:
        # Splits last dimension into odd and even dimensions
        x1, x2 = rearrange("... (half_d xy) -> xy ... half_d", x, xy=2)
        # Retrieve rotations corresponding to positions
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cls_cache, pos_ids)
        # Apply 2d rotation
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        # Recombine the rotated even and odd dimensions, interleaving the two elements:
        return rearrange("... half_d, ... half_d -> ... (half_d (1 + 1))", x1_rot, x2_rot).contiguous()


# uv run pytest -k test_scaled_dot_product_attention
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
    pdrop: float | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Scaled dot-product attention.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    d_k = K.shape[-1]
    qkt = einx.dot(" ... queries d_k, ... keys d_k -> ... queries keys", Q, K)
    attn_scores = qkt / np.sqrt(d_k)

    if mask is not None:  # Masking
        attn_scores = torch.where(mask, attn_scores, float('-Inf'))
    
    attn_weights = softmax(attn_scores, dim=-1)

    if pdrop is not None:  # Optional dropout
        attn_weights = nn.functional.dropout(attn_weights, pdrop)

    return einx.dot(" ... queries keys, ... keys d_v -> ... queries d_v", attn_weights, V)


# uv run pytest tests/test_model.py::test_multihead_self_attention
class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float = 0.) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop

        # Linear projection weights
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)  # Output projection back to d_model
        return

    def forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, " ... seq d_v"]:
        """
        """
        *b, seq_len, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension
        Q, K, V = (
            rearrange(" ... seq (heads d) -> ... heads seq d", X, heads=self.num_heads)
            for X in (Q, K, V)
        )

        # Construct causal mask
        # seq = torch.arange(seq_len, device=x.device)
        # # In einx, ellipses repeats the preceding expression
        # qi = rearrange("q -> b... 1 q 1", seq, b=[1]*len(b))
        # kj = rearrange("k -> b... 1 1 k", seq, b=[1]*len(b))
        # mask = (qi >= kj)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)

        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask, pdrop=self.attn_pdrop)

        # Concatenate attention output from all heads
        attn_output = rearrange("... heads seq d_v -> ... seq (heads d_v)", attn_output).contiguous()

        # Apply output projection
        return self.output_proj(attn_output)
    

# uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope
class CausalMultiheadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, positional_encoder: RotaryPositionalEmbedding, attn_pdrop: float = 0.) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.attn_pdrop = attn_pdrop

        # Linear projection weights
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)  # Output projection back to d_model

        self.positional_encoder = positional_encoder  # RoPE
        return

    def forward(self, x: Float[Tensor, "... seq d_model"], token_positions: Int[Tensor, "... seq"] | None = None) -> Float[Tensor, "... seq d_model"]:
        """
        """
        *b, seq_len, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension
        Q, K, V = (
            rearrange(" ... seq (heads d) -> ... heads seq d", X, heads=self.num_heads)
            for X in (Q, K, V)
        )

        if token_positions is None:
            token_positions = rearrange("seq -> b... seq", torch.arange(seq_len, device=x.device), b=[1] * len(b))

        # Duplicate token positions for each head
        token_positions = rearrange("... seq -> ... 1 seq", token_positions)
        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        # seq = torch.arange(seq_len, device=x.device)
        # qi = rearrange("q -> b ... 1 q 1", seq, b=[1]*len(b))
        # kj = rearrange("k -> b ... 1 1 k", seq, b=[1]*len(b))
        # mask = (qi >= kj)

        # Construct causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)

        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask, pdrop=self.attn_pdrop)

        # Concatenate attention output from all heads
        attn_output = rearrange("... heads seq d_v -> ... seq (heads d_v)", attn_output).contiguous()

        # Apply output projection
        return self.output_proj(attn_output)
    

# uv run pytest -k test_transformer_block 
class TransformerBlock(nn.Module):
    """
    A single Transformer layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, positional_encoder: RotaryPositionalEmbedding, attn_pdrop: float = 0., device: torch.device | None = None) -> None:
        super().__init__()
    
        self.attn = CausalMultiheadSelfAttentionWithRope(
            d_model=d_model, 
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            attn_pdrop=attn_pdrop,
        )
        self.ffn = SwiGLU(d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        return
        
    def forward(self, x: Tensor):
        """
        Implements pre-norm transformer.
        """
        z = x + self.attn(self.ln1(x))
        return z + self.ffn(self.ln2(z))


# uv run pytest -k test_transformer_lm
class BasicTransformerLM(nn.Module):
    """
    Transformer language model.
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float) -> None:

        # Record all input args in config
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Positional encoder
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, dim=d_model // num_heads, context_length=context_length)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.rope,
                )
                for _ in range(num_layers)
            ]
        ) 

        # Final processing
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)  # Output embedding

        logger.info(f"Number of non-embedding params: {self.get_num_params()/1e6:.2f}M")
        logger.info(f"Number of embedding params: {self.lm_head.weight.numel()/1e6:.2f}M")
    
    def get_num_params(self, non_embedding=True):
        """
        Returns the number of (non-embedding) parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.lm_head.weight.numel()
        return num_params

    def forward(self, x: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len vocab_size"]:
        """
        """
        x = self.token_embeddings(x)  # (batch_size, seq_len) -> (batch_size, seq_len, d_model)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)  # (batch_size, seq_len, d_model)

        # RMSNorm and head projection
        x = self.ln_final(x)  # (batch_size, seq_len, d_model)
        return self.lm_head(x)  # (batch_size, seq_len, vocab_size)

    def decode(self, prompt: Iterable, temperature: float, top_p: float, max_num_tokens: int | None = None):
        """
        Decode.
        """
        assert temperature >= 0
        assert 0 <= top_p <= 1

        return