"""Prepare sharded datasets to be uploaded to HF."""
from pathlib import Path

from datasets import load_dataset
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")
EOT = 50256
SHARD_SIZE = 100_000_000
TARGET_SHARDS = 12

out = Path("data")
out.mkdir(exist_ok=True)

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                   split="train", streaming=True)

buf, shard_idx, batch = [], 0, []
BATCH_DOCS = 1024

for doc in ds:
    batch.append(doc["text"])
    if len(batch) >= BATCH_DOCS:
        for toks in enc.encode_ordinary_batch(batch):
            buf.extend(toks)
            buf.append(EOT)
        batch = []
        while len(buf) >= SHARD_SIZE:
            arr = np.array(buf[:SHARD_SIZE], dtype=np.uint16)
            arr.tofile(out / f"shard_{shard_idx:04d}.bin")
            buf = buf[SHARD_SIZE:]
            shard_idx += 1
            print(f"Wrote shard {shard_idx}/{TARGET_SHARDS}")
            if shard_idx >= TARGET_SHARDS:
                break
    if shard_idx >= TARGET_SHARDS:
        break

print(f"Done: {shard_idx} shards")
