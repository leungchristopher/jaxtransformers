# JAX Transformers

## Goals of this project:
- `gpt-jax/`: Python/Jax training of a transformer model with proper MFU accounting.
- `gpt-rs/`: Rust inference engine which loads safetensors exported from JAX, runs the forward pass on GPU via `cudarc` (raw CUDA driver bindings). Implementations of KV-cache, top-k/temperature sampling and a CLI.
- `kernels/`: CUDA C++ kernels invoked from both sides. JAX will call them via `jax.ffi`/`jax_triton` to CUDA, Rust will call them via `cudarc`

# Implementation progress
20/04: dataset sharding, environment setup, start of model development