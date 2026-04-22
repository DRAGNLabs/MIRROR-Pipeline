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
      → Preprocessors (convert raw data into model-compatible format, e.g. by tokenizing and padding)
      → Models (compute forward passes)
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

The `fit()` method is the most important part of the class. It accepts a model, dataset, preprocessor, optional checkpoint, and training hyperparameters (epochs, batch size, etc.), as well as optional validation and test datasets. It begins by setting up the model and optimizer with Fabric, then builds the training dataloader (and optionally validation/test dataloaders). If a checkpoint is provided, it resumes training from that checkpoint. 

The core loop iterates over epochs and batches: for each batch it zeroes gradients, calls `model.training_step(batch)` to get the loss, backpropagates via Fabric, and steps the optimizer.

### Datasets

Datasets provide a unified interface for loading data, such as text rows, or other types of data (e.g., the [MCQA repository](https://github.com/DRAGNLabs/MIRROR-MCQA-Decisions) uses McqaRows). They all extend `MirrorDataset`, a generic typed class built on top of PyTorch's `Dataset`.

Available datasets include the HuggingFace datasets `ImdbDataset` and `WikitextDataset`, which must be downloaded once using HuggingFace credentials, and thereafter will be automatically cached locally. `TxtDataset` allows plain text files, such as the Church Text Dataset, to be used as datasets as well. `OnDemandPreprocessedDataset` is a wrapper that supports "lazily" preprocessing a dataset on-the-fly rather than upfront (useful for memory efficiency).

### Preprocessors

Preprocessors convert raw data into a format suitable for model training. They extend `MirrorPreprocessor`, which defines two methods: `preprocess_example(example)` (usually converts a single raw example to token IDs) and `collate(examples)` (batches processed examples together into a single object).

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

## Common Commands

### SSH

- `ssh <username>@ssh.rc.byu.edu`: Connect to the BYU supercomputer via SSH
    - `<username>` is your BYU Net ID
    - You will be prompted for your password and Duo 2FA

- `scp <filename> <username>@ssh.rc.byu.edu:<path/to/destination>`: Copy a file from your local machine to the supercomputer
    - Use `scp <username>@ssh.rc.byu.edu:<path/to/file> .` to copy a file from the supercomputer to your current local directory

- `ssh-keygen`: Generate an SSH key pair so you don't have to enter your password every time
    - After generating, copy your public key to the supercomputer with `ssh-copy-id <username>@ssh.rc.byu.edu`

- `exit`: Disconnect from the supercomputer and return to your local terminal

### On Startup

- `source /etc/profile`: Load the system-wide environment settings and paths
    - If you're logging in directly through ssh in the terminal, this will be done for you. Otherwise (e.g. using VSCode's UI), run this first every time you log in so that tools like `mamba` and other modules are available

- `mamba activate ./.env`: Activate the project's conda environment
    - This loads the Python version and dependencies needed for the MIRROR Pipeline
    - Must be run in the MIRROR Pipeline directory

### Terminal

- `cd <directory-name>`: Change into a directory
    - Use `cd ..` to go up/back one directory, or `cd ~` to go to your home directory

- `ls`: List files and folders in the current directory

- `mkdir <directory-name>`: Create a new directory
    - Use `mkdir -p <path/to/nested/directory>` to create nested directories all at once

- `rm <filename>`: Delete a file
    - Use `rm -r <directory-name>` to delete a directory and everything inside it

- `cp <source> <destination>`: Copy a file or directory to a new location
    - Use `cp -r` to copy a directory and its contents

- `mv <source> <destination>`: Move or rename a file or directory
    - Use `mv <old-name.py> <new-name.py>` to rename a file

- `cat <filename>`: Print the contents of a file to the terminal
    - Useful for quickly viewing small files

- `clear`: Clear the terminal screen

### Git

- `git status`: Show the current state of your working directory and staging area
    - Lists which files are staged, modified, or untracked so you know what will be included in your next commit

- `git add`: Stage files that have been changed so they are included in the next commit
    - Use `--all` to add all changes (excluding untracked files, e.g. those in the `.gitignore`), or `<filename>` to add a specific file

- `git commit -m "<message>"`: Take staged changes and commit them, with a comment explaining what changed in the commit. 
    - You can also run `git commit -am "<message>"` to stage all modified tracked files and commit them at the same time 

- `git push`: Push the commit to the repository

- `git pull`: Pull the latest changes from the remote repository and merge them into your current branch
    - Essentially `git fetch` and `git merge` in one step
    - Use `git pull origin main` to pull changes from the main branch into your current branch

- `git fetch origin`: Download the latest changes from the remote repository without merging them into your current branch
    - Useful when you want to see what changes are available before merging

- `git merge`: Merge changes from one branch into your current branch
    - Use `git merge main` to merge the main branch into whatever branch you're currently on
    - If there are conflicts, Git will ask you to resolve them before completing the merge
    
- `git checkout <branch-name>`: Switch to a different branch 

### Vim

Vim is the default editor for commit message files (e.g. git merge).

- `vim <filename>`: Open a file in Vim
    - If the file doesn't exist, Vim will create it when you save

- `i`: Enter Insert mode so you can type and edit text
    - Press `Esc` to go back to Normal mode when you're done typing

- `:w`: Save the current file (write)
    - Use `:w <filename>` to save as a specific filename

- `:q`: Quit Vim
    - Use `:q!` to quit without saving changes
    - Use `:wq` to save and quit at the same time
    
### Slurm

- `squeue -u $USER`: Check the status of your submitted jobs
    - Shows job ID, status (pending, running), time elapsed, and which node it's running on

- `scancel <job-id>`: Cancel a running or pending job
    - Use `scancel -u $USER` to cancel all of your jobs at once

- `sinfo`: View available partitions and node status
    - Useful for seeing which partitions have available resources

### MIRROR Pipeline

- `python src/main.py fit --config <config-file>`: Train a model using settings from a config file
    - The config file specifies the dataset, model, preprocessor, training parameters, and SLURM settings
        - Config files should be local and user-specific; start your config file(s) names with "config" so that the `.gitignore` knows not to track them
    - You can also pass arguments directly, e.g. `python src/main.py fit --data.class_path WikitextDataset --model.class_path MirrorLlamaModel --epochs 1 --batch_size 1`

- `python src/main.py preprocess --config <config-file>`: Preprocess a dataset without training
    - Useful for preparing data separately before running a training job
    - Requires `--data` and `--preprocessor` to be specified (either in the config file or as command-line arguments)

- `python src/launch_jupyter.py`: Set up a Jupyter server on a compute node for running training jobs 
    - Jupyter notebooks allow for significantly decreased startup time on repeat job runs
    - This command will output a URL, which is used to set the environment for `jupyter_template.ipynb` (or your copy(s) of it)

### Pre-Pull Request Checklist
[specific test runs, formatting checks they must run locally before opening a Pull Request]
 
### New Ticket Kickoff Checklist

#### Starting work on an existing ticket

1. Assign the ticket to yourself
    Click the gear icon next to "Assignees" in the top right and select your GitHub profile.

2. Mark ticket as In Progress
    Below the Assignees section, in the Projects section, the ticket will have a blue "Ready" status tag. Click on it and select the orange "In Progress" tag.

3. Create a new branch
    Near the bottom of the right side bar, under "Development", click the "Create a branch" link. Click the green "Create branch" button in the pop-up, and copy the git commands that pop up. Run these commands in your terminal; this will switch you to your new branch.

#### Creating a new ticket

1. Create the ticket/issue

    In the "Backlog" column of the project pipeline, press the `+` button to add an item. In the text bar that opens up at the bottom, enter a descriptive title for the new ticket, and hit Enter. Select the correct repository, and press the "Blank issue" button.

2. Give the ticket a description

    Unless the ticket is very simple and its title is self-explanatory, give a brief explanation of the issue or feature this ticket is trying to fix or implement. Add any details about potential implementations you've thought of, other tickets this may be related to, etc. 

3. Acceptance criteria

    Add "Acceptance criteria: " to the bottom of the ticket description, then a bullet point for each criterion or condition that should be met for this ticket to be considered complete.

4. Hit "Create"

    Hit the green "Create" button to finish creating the ticket. If the ticket is of higher priority or relevance, you can move it immediately from Backlog to Ready (or In Progress if you plan to immediately work on this ticket, e.g. if you're trying to fix an emergency bug).  
