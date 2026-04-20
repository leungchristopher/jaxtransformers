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

if __name__ == "__main__":
    norm = RMSNorm(8, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 4, 8)) * 3.0
    y = norm(x)
    print(y.shape, y[0, 0, 0])
