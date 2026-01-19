## Instructions

1. Create a local mamba environment `mamba create --yes -f environment.yml -p ./.env`
  a. If at some point you need to update your local environment run `mamba env update --file environment.yml --prune -p ./.env` 
2. Activate the environment `mamba activate ./.env`
3. (Optional) deactivate the environment `mamba deactivate`
4. Run using `launch.sh`. Modify `launch.sh` as needed. Later we will have a better way to launch.
  a. Example using callbacks: `python src/main.py fit --callbacks='[{"class_path": "mirror.callbacks.checkpoint_callback.CheckpointCallback", "init_args": {"every_n_train_steps": 3}}]'`

