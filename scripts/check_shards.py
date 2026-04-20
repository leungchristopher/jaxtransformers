"""Check shards"""
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
arr = np.fromfile("data/shard_0000.bin", dtype=np.uint16)
print(f"shard 0: {len(arr):,} tokens")
print(enc.decode(arr[10000:10200].tolist()))
