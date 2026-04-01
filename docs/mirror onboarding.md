# MIRROR Onboarding

## General Pipeline Information

### Initial Access Setup

 [Add Content Here]

### Things to do every time you log in

[Add Content Here]
 
## General MIRROR Pipeline Architecture

The MIRROR pipeline follows a modular architecture where each component handles a distinct responsibility. The overall data flow is:

```
main.py (CLI entry point)
  → subcommands.py (routes to pipelines such as `fit` or `preprocess`)
    → trainer.py (orchestrates the training loop)
      → Datasets (load raw data, e.g. text)
      → Preprocessors (tokenize data into batches)
      → Models (compute forward/backward passes)
      → Callbacks (handle checkpointing, logging, progress, etc.)
```

On a login node, the pipeline downloads required assets (like tokenizers/models) and submits an SBATCH training job; on a compute node, it executes the actual training.

### main.py

This is the entry point for the pipeline. It parses command-line arguments/YAML config files, then routes to the appropriate subcommand.

A first parse determines which subcommand was invoked. `jsonargparse` is then used to load arguments from a YAML config file and instantiate the appropriate `MirrorModel`, `MirrorPreprocessor`, etc. On login nodes, the script downloads the model/tokenizer and submits the job to a compute node. On compute nodes, it calls `fit()` directly.

An example YAML config can be found in [config-example.md](config-example.md).

### subcommands.py

This module implements the pipeline's subcommands. The two main subcommands:
- `fit()` — The main training subcommand. If running on a login node with `job_type=compute`, it submits a training job to the supercomputer; otherwise (i.e. either we're training locally/on a login node, or we're already on a compute node), it executes training directly via `trainer.fit()`.
- `preprocess()` — Applies a preprocessor to a dataset and caches the result, without running training.

The `templates/` directory contains `slurm.jinja`, a Jinja2 SBATCH template. When submitting a job from a login node, `subcommands.py` fills in this template and passes the result to `sbatch` to submit a training job to the supercomputer.

### trainer.py

The `Trainer` class drives the training loop, built on top of PyTorch Lightning Fabric to support training distributed across nodes/GPUs.

**Key methods:**
- `launch()` — Initializes the Fabric launcher.
- `fit()` — Runs the full training loop: sets up the model and optimizer with Fabric, optionally loads a checkpoint, creates dataloaders, and iterates over epochs and batches. Calls Callbacks at each stage.
- `_eval_loop()` — Runs validation or test evaluation with gradients disabled.
- `_make_dataloader()` — Creates a PyTorch DataLoader, optionally wrapping the dataset with `OnDemandPreprocessedDataset` for lazy preprocessing.

### Datasets

Datasets provide a unified interface for loading data, such as text rows, or other types of data (e.g., the [MCQA repository](https://github.com/DRAGNLabs/MIRROR-MCQA-Decisions) uses McqaRows). They all extend `MirrorDataset`, a generic typed class built on top of PyTorch's `Dataset`.

Available datasets include the HuggingFace datasets `ImdbDataset` and `WikitextDataset`, which must be downloaded once using HuggingFace credentials, and thereafter will be automatically cached locally. `TxtDataset` allows plain text files, such as the Church Text Dataset, to be used as datasets as well. `OnDemandPreprocessedDataset` is a wrapper that supports "lazily" preprocessing a dataset on-the-fly rather than upfront (useful for memory efficiency).

### Preprocessors

Preprocessors convert raw data into tokenized batches suitable for model training. They extend `MirrorPreprocessor`, which defines two methods: `preprocess_example(example)` (usually converts a single raw example to token IDs) and `collate(examples)` (batches processed examples together into a single object).

Each model has its own preprocessor because each model architecture uses a distinct tokenizer and vocab size. For example, GPT-2 has `vocab_size = 50257` while Llama 3.2-1B has `vocab_size = 128256`, so they each have their own preprocessor with the correct vocab size. However, preprocessors can be mixed and matched as long as the vocab sizes match — for example, you could use a custom Llama config with `vocab_size = 50257` and pair it with `MirrorGPTPreprocessor` instead.

Tokenizers are downloaded/cached just like models and datasets.

### Models

This is where specific model implementations lie. Each extends the `MirrorModel` class, requiring a `preprocessor` property, a `training_step(batch)` method that returns a loss, and a `configure_optimizers()` method.

Models are downloaded and cached similarly to datasets.

The models module also includes a `Whitebox Transformers` subsystem, which provides a type-safe way to extract optional outputs (loss, hidden states, attentions) from HuggingFace models by chaining method calls together (e.g., `.fresh(model).include_loss(labels).execute(batch)`).

### Callbacks

Callbacks allow Fabric to execute custom functions at specific points during the training process. Each callback implementation extends the `Callback` base class, which defines methods like `on_fit_start`, `on_fit_end`, `on_train_batch_end`, `on_validation_epoch_end`, and `on_test_epoch_end`. For example, `self.fabric.call("on_fit_start", ...)` will call the `on_fit_start` method with the provided parameters on each callback that implements that method.

Essentially, callbacks control "side-effects" that don't directly affect the training process — for example, printing training progress bars.

### Interventions

Interventions are model wrappers — MirrorModels that contain a MirrorModel — that allow for modifications to the architecture of base models, such as mirror neuron capabilities. In contrast to callbacks, which don't directly affect training, interventions are for directly altering the structure or behavior of the model/training process.

Future model interventions will be implemented in this module.

## Pipeline Developer Information

### Pre-Pull Request Checklist
[specific test runs, formatting checks they must run locally before opening a Pull Request]

### Ticket 5: New Ticket Kickoff Checklist
[administrative steps required for assigning yourself a new ticket (move the card, assign yourself, create a branch, etc.), and also for writing a new ticket]