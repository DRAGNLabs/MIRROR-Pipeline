# Folder Structure of ~/nobackup/autodelete/mirror_data

### Quick reference:
- checkpoints: found inside `training_runs`
- config.json, generation_config.json, model.safetensors: found inside `models` / `[model_family]` / `[model_name]`
- tokenizer.json, tokenizer_config.json: found inside `tokenizers` / `[model_family]` / `[tokenizer_name]`



### Folder Structure
```text
mirror_data/
|-- datasets/
|   `-- Salesforce/
|       `-- wikitext/
|-- hf_cache/
|-- models/
|   |-- meta-llama/
|   |   |-- Llama-3.2-1B/
|   |   `-- Llama-3.2-1B-Instruct/
|   `-- openai-community/
|       `-- gpt2/
|-- requeue_handoffs/
|-- tokenizers/
|   |-- meta-llama/
|   |   |-- Llama-3.2-1B/
|   |   `-- Llama-3.2-1B-Instruct/
|   `-- openai-community/
|       `-- gpt2/
`-- training_runs/
```

#### `datasets`
Contains dataset source folders (e.g. Salesforce), which contain dataset folders (e.g. wikitext).

#### `hf-cache`
Contains cached models/tokenizers from huggingface_hub, using its cached file format. 

#### `models`
Contains model family folders (e.g. meta-llama), which contain specific model folders (e.g. Llama-3.2-1B-Instruct). Each model folder contains the model's config.json, generation_config.json, and model.safetensors files.

#### `requeue_handoffs`
Contains bookkeeping for SLURM job requeuing/resuming.

#### `tokenizers`
Contains model family folders (e.g. meta-llama), which contain specific model folders (e.g Llama-3.2-1B-Instruct). Each model folder contains the model's tokenizer.json and tokenizer_config.json files.

#### `training_runs`
Contains a folder with logs for each submitted training run, including checkpoints.

#### `wandb`
Contains Wandb data saved offline, which can be synced to the cloud from a login node.
