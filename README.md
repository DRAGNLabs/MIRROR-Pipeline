## Instructions

1. Create a local mamba environment `mamba create --yes -f environment.yml -p ./.env`
1. Run using `launch.sh`. Modify `launch.sh` as needed. Later we will have a better way to launch.
  1. Example using callbacks: `python src/main.py fit --callbacks='[{"class_path": "mirror.callbacks.checkpoint_callback.CheckpointCallback", "init_args": {"every_n_train_steps": 3}}]'`

