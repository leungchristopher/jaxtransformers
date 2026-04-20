# justfile
set allow-duplicate-variables
set allow-duplicate-recipes

# Default project variables
hf-user := "default-organization" 
dataset := "gpt-jr-fineweb-edu-1b"

# upload tokenized shards to HF 
upload-data:
    huggingface-cli upload {{hf-user}}/{{dataset}} data/ . --repo-type dataset

# download tokenized shards from HF
download-data:
    huggingface-cli download {{hf-user}}/{{dataset}} \
      --repo-type dataset \
      --local-dir /workspace/data

# Include local overrides
-include .justfile.local