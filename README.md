## Instructions

1. Create a python environment with python version >= 3.12.11
2. Create a venv `python -v venv env`
3. Activate the venv `env/bin/activate`
4. Install requirements `pip install -r requirements.txt`
5. (Optional) deactivate the venv `deactivate`
6. Run using `launch.sh`. Modify `launch.sh` as needed. Later we will have a better way to launch.
  1. Example using callbacks: `python src/main.py fit --callbacks='[{"class_path": "mirror.callbacks.checkpoint_callback.CheckpointCallback", "init_args": {"every_n_train_steps": 3}}]'`

