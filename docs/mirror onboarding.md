# MIRROR Onboarding

## General Pipeline Information

### Initial Access Setup

 [Add Content Here]

### Things to do every time you log in

[Add Content Here]
 
## General MIRROR Pipeline Architecture

### Core Components

The MIRROR pipeline follows a modular architecture where each component handles a distinct responsibility. The overall data flow is:

```
main.py (CLI entry point)
  → subcommands.py (routes to fit/preprocess)
    → trainer.py (orchestrates the training loop)
      → Datasets (load raw text)
      → Preprocessors (tokenize text into batches)
      → Models (compute forward/backward passes)
      → Callbacks (handle checkpointing, logging, progress, etc.)
```

On SLURM clusters, the pipeline detects whether it is running on a login node or a compute node. On a login node it downloads required assets (like tokenizers/models) and submits an SBATCH job; on a compute node it executes the actual training.

#### Models

This is where specific model implementations lie. Each extends the `MirrorModel` class, requiring a `preprocessor` property, a `training_step(batch)` method that returns a loss and a `configure_optimizers()` method.

Models are downloaded and cached similarly to datasets.

The models module also includes a `Whitebox Transformers` subsystem, which provides a type-safe way to extract optional outputs (loss, hidden states, attentions) from HuggingFace models by chaining method calls together (e.g., `.fresh(model).include_loss(labels).execute(batch)`).

#### Datasets

Datasets provide a unified interface for loading data, such as text rows, or other tpyes of data (e.g., the MCQA repository uses McqaRows). They all extend `MirrorDataset`, a generic typed class built on top of PyTorch's `Dataset`. 

Available datasets include the HuggingFace datasets `ImdbDataset` and `WikitextDataset`, which must be downloaded once using HuggingFace credentials, and thereafter will be automatically cached locally. `TxtDataset` allows plain text files, such as the Church Text Dataset, to be used as datasets as well. `OnDemandPreprocessedDataset` is a wrapper that supports "lazily" preprocessing a dataset on-the-fly rather than upfront (useful for memory efficiency).

#### Preprocessors

Preprocessors convert raw data into tokenized batches suitable for model training. They extend `MirrorPreprocessor`, which defines two methods: `preprocess_example(example)` (usually converts a single raw example to token IDs) and `collate(examples)` (batches processed examples together into a single object; usually combines token tensors into a batched tensor with padding). Available preprocessors include GPT2's `MirrorGPTPreprocessor`, Llama-3.2-1B's `MirrorLlamaPreprocessor`, and our custom `BabblePreprocessor` designed for use with the Church Text Dataset. Preprocessors can be mixed and matched with models, but the vocab sizes must match. For example, to use `MirrorLlamaModel` with `MirrorGPTPreprocessor`, since the GPT preprocessor's vocab size is 50257, you must use a custom config of `MirrorLlamaModel` with the vocab size set to 50257, rather than its default of 128256.

Tokenizers are downloaded/cached just like models and datasets.

#### Callbacks

Callbacks allow Fabric to execute custom functions at specific points during the training process. Each callback implementation extends the `Callback` base class, which defines methods like `on_fit_start`, `on_fit_end`, `on_train_batch_end`, `on_validation_epoch_end`, and `on_test_epoch_end`. For example, `self.fabric.call("on_fit_start", ...)` will call the `on_fit_start` method with the provided parameters on each callback that implements that method.

#### Interventions

Interventions allow for modifications to the architecture of base models - e.g., mirror neuron capabilities. Future model interventions will be implemented in this module.

#### Templates

The templates directory contains `slurm.jinja`, a Jinja2 template for generating SBATCH submission scripts to send training jobs to the supercomputer. It accepts variables for SLURM directives (time, nodes, ntasks-per-node, gpus-per-node, mem-per-cpu, output path, signal handling), plus options like `--requeue` and `--qos`.

This template is rendered by `subcommands.py` when submitting jobs from a login node.

#### subcommands.py

This module implements the CLI subcommands that the pipeline supports. 

The two main subcommands: 
- `fit()` — The main training subcommand. Accepts configuration for data, model, trainer, preprocessor, checkpoint, SLURM settings, epochs, batch size, and more. If running on a login node with `job_type=compute`, it submits an SBATCH job via `_submit_slurm_job()`. Otherwise it executes training directly via `trainer.fit()`. 
- `preprocess()` — A standalone preprocessing subcommand that applies a preprocessor to a dataset and caches the result, without running training.
- `_submit_slurm_job()` — An internal helper that renders the `slurm.jinja` template with the current configuration, submits the job via the `sbatch` command, and returns the job ID. It prevents infinite recursion by stripping the `--slurm.submit` flag from the resubmitted command.

#### trainer.py

The Trainer class is the core training loop orchestrator, built on top of PyTorch Lightning Fabric for distributed training support. It is generic and type-parameterized.

**Constructor parameters:**
- `strategy` — Lightning Strategy (defaults to `FSDPStrategy` for distributed training)
- `devices` — Number of gpus per node
- `num_nodes` — Number of compute nodes
- `callbacks` — List of Callback instances

**Key methods:**
- `launch()` — Initializes the Fabric launcher. Falls back to CPU with `SingleDeviceStrategy` if CUDA is unavailable.
- `fit()` — Runs the full training loop: sets up the model and optimizer with Fabric, optionally loads a checkpoint, creates dataloaders, and iterates over epochs and batches. Calls callback hooks at each stage.
- `_eval_loop()` — Runs validation or test evaluation with gradients disabled.
- `_make_dataloader()` — Creates a PyTorch DataLoader, optionally wrapping the dataset with `OnDemandPreprocessedDataset` for lazy preprocessing.

The Trainer deduplicates singleton callbacks to prevent multiple instances of the same callback type.

#### main.py

This is the CLI entry point for the entire pipeline. It parses command-line arguments and YAML config files, then dispatches to the appropriate subcommand.

**Supported subcommands:** `fit`, `test`, `preprocess`

**Argument parsing flow:**
1. A first parse determines which subcommand was invoked.
2. A second, subcommand-specific parse uses `jsonargparse` to build an `ArgumentParser` that supports `--config` for YAML config files. It adds function-level arguments for the subcommand and registers subclass types for `MirrorModel`, `MirrorDataset`, `MirrorPreprocessor`, etc., so they can be instantiated from config.
3. `init_config(device)` is called to detect the runtime environment (local, SLURM login, or SLURM compute).
4. On login nodes with a compute job type, it downloads the model and tokenizer, then returns (the actual training happens after SBATCH submission). On compute nodes, it calls `fit()` to execute training.

An example YAML config can be found in `[[config-example.md]]`. 

## Pipeline Developer Information

### Pre-Pull Request Checklist
[specific test runs, formatting checks they must run locally before opening a Pull Request]

### Ticket 5: New Ticket Kickoff Checklist
[administrative steps required for assigning yourself a new ticket (move the card, assign yourself, create a branch, etc.), and also for writing a new ticket]