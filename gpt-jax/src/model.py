"""JAX implementation of GPT-2"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

@dataclass(frozen=True)  # Ensures instances are hashable+mutable
class GPTConfig:
    """Config matches GPT-2"""
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_head: int = 64
    d_mlp: int = 2048
    ctx_len: int = 1024

def __post_init__(self):
    assert self.n_head * self.d_head == self.d_model, \
        f"n_head * d_head ({self.n_head * self.d_head}) must equal d_model ({self.d_model})"

class RMSNorm(nnx.Module):
    """y_i = x_i / RMS(x) * gamma_i"""
    def __init__(self, d: int, *, eps: float = 1e-6, rngs:nnx.Rngs):
        self.weight = nnx.Param(jnp.ones((d,)))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (..., d)
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return x * self.weight[...]

class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        self.wq = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=False, rngs=rngs)
        self.wk = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=False, rngs=rngs)
        self.wv = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        B, T, D = x.shape
        H, Dh = self.n_head, self.d_head

        # Q, K, V projections, splitting (..., D) -> (..., H, Dh)
        q = self.wq(x).reshape(B, T, H, Dh)
        k = self.wk(x).reshape(B, T, H, Dh)
        v = self.wv(x).reshape(B, T, H, Dh)

        # Attn scores: (B, H, T, T)
        # scaling factor prevents memory use blowup
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(Dh).astype(x.dtype)

        # Causal mask
        if mask is None:
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        attn = jax.nn.softmax(scores, axis=-1)
        
        out = jnp.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, D)
        
        return self.wo(out)


class MLP(nnx.Module):
    """
    MLP layer uses SwiGLU (Shazeer demonstrated it outperformed regular GLU
    Parallel projections split the expansion step, but adds 50% more params.
    Compromise: reduce d_mlp.
    """
    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.w1 = nnx.Linear(cfg.d_model, cfg.d_mlp, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(cfg.d_mlp, cfg.d_model, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(cfg.d_model, cfg.d_mlp, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))

class Block(nnx.Module):
    """
    Adopt prenorm, as this stops the residual stream from being normalised repeatedly.
    This should increase performance in depth.
    """
    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.norm1 = RMSNorm(cfg.d_model, rngs=rngs)
        self.attn = CausalSelfAttention(cfg, rngs=rngs)
        self.norm2 = RMSNorm(cfg.d_model, rngs=rngs)
        self.mlp = MLP(cfg, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nnx.Module):
    """
    Use nnx.Embed for tables indexed by integer id - gather op faster than one-hot matmul
    use .attend(x), which does (x @ embedding.T), to halve embedding/unembedding cost
    """
    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.wte = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        self.wpe = nnx.Embed(cfg.ctx_len, cfg.d_model, rngs=rngs)
        self.blocks = nnx.List([Block(cfg, rngs=rngs) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, rngs=rngs)
    
    def __call__(self, idx: jax.Array) -> jax.Array:
        # idx (B, T) int32 token ids
        B, T = idx.shape
        assert T <= self.cfg.ctx_len, f"sequence length {T} > ctx_len {self.cfg.ctx_len}"
        
        pos = jnp.arange(T)[None, :]  # (1, T)
        x = self.wte(idx) + self.wpe(pos)  # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.wte.attend(x)  # (B, T, vocab)
        return logits

def count_params(model: nnx.Module) -> int:
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree.leaves(state)
    return sum(int(x.size) for x in leaves)


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg, rngs=nnx.Rngs(0))
    n = count_params(model)
    print(f"parameters: {n:,}  ({n / 1e6:.2f}M)")

    # Forward-pass shape check.
    idx = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = model(idx)
    print(f"logits shape: {logits.shape}")  # (2, 16, 50257)
    assert logits.shape == (2, 16, cfg.vocab_size)
    print("forward pass OK")