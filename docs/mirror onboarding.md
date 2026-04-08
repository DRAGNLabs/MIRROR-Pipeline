# MIRROR Onboarding

## General Pipeline Information

### Initial Access Setup

 [Add Content Here]

### Things to do every time you log in

[Add Content Here]
 
### General MIRROR Pipeline Architecture

[the core components, data flow, and primary services that make up the MIRROR Pipeline]

[how to use the MIRROR Pipeline. This will essentially be the documentation for our pipeline]
 
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

## How to use the MIRROR Pipeline

It's highly recommended to use YAML config files rather than passing in arguments through the command line. Thus, this section will outline how to use the MIRROR Pipeline through the lens of creating this config file.

First, select your model:
```yaml
model:
    class_path: MirrorLlamaModel # or MirrorGPTModel, etc
    init_args: # If using Llama: 
        initialization:
            # Pretrained weights:
            <model_spec> # 3.2-1B | 3.2-1B-Instruct
            # OR custom config:
            vocab_size: <vocab_size>
            hidden_size: <hidden_size> # etc

        # If using GPT2:
        weights: <weights> # pretrained | random
    # Etc/as appropriate for other models
```

Next, decide how the data will be preprocessed. The model's own preprocessor will be used by default, but it can be overridden (see Mirror Architecture/Preprocessors):

```yaml
preprocessor:
    class_path: <class_path> # e.g. MirrorLlamaPreprocessor
```

To preprocess up front rather than on-the-fly:

```yaml
do_preprocess: True # False by default
``` 

Now pick your dataset and how it will be used:

```yaml
data: # Training set 
  class_path: WikitextDataset
  init_args:
    split: train # Use the dataset's `train` portion
    skip: <i> # Optionally, skip the first <i> examples from the selected split 
    head: <j> # Use the first <j> examples (after skipping)
    # OR
    # start_fraction: 0.0
    # end_fraction: 0.8 # Use the first 80% of the selected dataset/split

val_data: # Validation set
  class_path: WikitextDataset
  init_args:
    split: validation # Use the dataset's `validation` portion
    ... # etc

val_check_interval: <k> # Perform validation every <k> epochs

test_data: # Test set
  class_path: WikitextDataset
  init_args:
    split: test # Use the dataset's `test` portion
    ... # etc
```

Next, pick your SLURM job/training run settings. 

```yaml
slurm:
  job_type: "compute" # "local" "local-download"
  time: "01:00:00" # Timeout limit - note that the requeue callback will continue the job if the time limit is reached by submitting another SLURM job
  gpus_per_node: <gpu_type>:<num_gpus> # GPUs, low -> high capability: p100 | a100 (dw87 cluster) | h200  
  nodes: 1 # Number of nodes to allocate
  mem_per_cpu: "128G" # Memory allocated per CPU core
  output: "slurm_logs/%j.out" # Path for job stdout log (%j = job ID)
  open_mode: "append" # Append to log file instead of overwriting
  signal: "SIGHUP@90" # Send SIGHUP signal 90 seconds before timeout to allow graceful shutdown
  requeue: true # Automatically requeue job if it is preempted
  qos: <cluster> # Specific GPU cluster/config, e.g. dw87, test

epochs: 1 # Num of passes through the training dataset
batch_size: 1 # Num of samples in each batch 
device: cuda # `cuda` for GPU, `cpu` for CPU
```

Now submit the training job by running the following command:

`python src/main.py fit --config <configfilename>.yaml`

Replace `fit` with `preprocess` if you are just trying to do a preprocessing run. It's smart to make a separate config file for preprocessing, which won't need parameters like `model`, `val_data`, `test_data`, etc. That way, instead of constantly editing your main config file, you can just pass in your preprocessing config file for preprocessing runs.


### Pre-Pull Request Checklist
[specific test runs, formatting checks they must run locally before opening a Pull Request]
 
### Ticket 5: New Ticket Kickoff Checklist
[administrative steps required for assigning yourself a new ticket (move the card, assign yourself, create a branch, etc.), and also for writing a new ticket]
