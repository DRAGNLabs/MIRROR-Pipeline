**Example YAML config**

Note that each class' sub-arguments in a config file are determined by that class' constructor parameters.

```yaml
data:                           # Training dataset (required)
  class_path: WikitextDataset   # Dataset class: WikitextDataset | ImdbDataset | TxtDataset (doesn't support split)
  init_args:
    split: train                # Which split to use: "train" | "validation" | "test"
    head: 100                   # Max examples to load (null = all)
    skip: null                  # Examples to skip from the start (null = none)

val_data:                       # Validation dataset (optional)
  class_path: WikitextDataset
  init_args:
    split: validation
    head: 10
    skip: null

test_data:                      # Test dataset (optional)
  class_path: WikitextDataset
  init_args:
    split: test
    head: 10
    skip: null

val_check_interval: 1.0         # Run validation every N epochs (1.0 = every epoch)

# For TxtDataset, use file_path instead of split:
# data:
#   class_path: TxtDataset
#   init_args:
#     file_path: data/my_corpus.txt  # Path to a plain-text file (one example per line)
#     head: null

do_preprocess: true     # Preprocess the whole dataset upfront (true) vs. on-the-fly (false)

preprocessor:      # Tokenizer/preprocessor (optional; auto-selected from model if omitted)
  class_path: MirrorLlamaPreprocessor  # MirrorLlamaPreprocessor | MirrorGPTPreprocessor

model:
  class_path: mirror.models.mirror_llama_model.MirrorLlamaModel  # mirror.models.mirror_gpt_model.MirrorGPTModel
  init_args:
    # Option 1 – pretrained preset:
    # initialization: "3.2-1B-Instruct"   # Preset: "3.2-1B" | "3.2-1B-Instruct"

    # Option 2 – custom architecture:
    initialization:
      vocab_size: 128256             # Number of tokens in the vocabulary
      hidden_size: 512               # Model embedding / hidden dimension
      intermediate_size: 1024        # Feed-forward layer inner dimension
      num_hidden_layers: 8           # Number of transformer blocks
      num_attention_heads: 8         # Number of attention heads
      num_key_value_heads: null      # KV heads for grouped-query attention (null = same as num_attention_heads)
      tie_word_embeddings: false     # Share input/output embedding weights

# For MirrorGPTModel, the init_args are:
# model:
#   class_path: mirror.models.mirror_gpt_model.MirrorGPTModel
#   init_args:
#     weights: "pretrained"          # "pretrained" (load GPT-2 weights) | "random"


trainer:
  devices: 1                       # GPUs/devices per node
  num_nodes: 1                     # Number of compute nodes
  # strategy is auto-selected (FSDP for multi-GPU, SingleDevice otherwise)
  callbacks:
    - class_path: CheckpointCallback     # Save model checkpoints
      init_args:
        every_n_train_steps: null        # Save every N steps (null = only at start and end)
    - class_path: WandbCallback          # Customize Wandb output
      init_args:
        log_every_n_steps: 1             # Log train metrics every N training steps
        extra_metrics_getter:
          class_path: GradNormMetrics
    - class_path: ConfigSnapshotCallback # Snapshot the config file alongside each checkpoint
    - class_path: ProgressCallback       # Print live loss / progress to stdout

# checkpoint:
#   training_run_id: "20240101_120000"   # ID of the run to resume from
#   checkpoint_name: "start"               # Checkpoint to load: "start" | "end" | step number e.g. "0100"

slurm:
  job_type: "compute"     # "compute" (submit to SLURM) | "local" | "local-download" (download only)
  time: "01:00:00"        # Wall-clock time limit in HH:MM:SS
  gpus_per_node: p100:1   # GPU type and count per node, e.g. "p100:1" | "a100:4" | "h200:2"
  nodes: null             # Node count (null = auto-inferred from trainer.num_nodes)
  ntasks_per_node: null   # MPI tasks per node (null = auto-inferred from trainer.devices)
  mem_per_cpu: "128G"     # Memory per CPU core
  output: "slurm_logs/%j.out" # Log file path (%j = SLURM job ID)
  open_mode: "append"     # Log file open mode: "append" | "truncate"
  signal: "SIGHUP@90"     # Signal sent before timeout to trigger graceful shutdown/requeue
  requeue: true           # Automatically requeue the job if it is preempted
  qos: null               # Quality-of-Service tier (null = cluster default)


epochs: 1         # Total training epochs
batch_size: 1     # Examples per gradient-update step
device: cpu       # Compute device: "cpu" | "cuda" (auto-detected on SLURM compute nodes)
```

