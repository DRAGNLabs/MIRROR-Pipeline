## Instructions

1. Create a local mamba environment `mamba create --yes -f environment.yml -p ./.env`
    - If at some point you need to update your local environment run `mamba env update --file environment.yml --prune -p ./.env` 
2. Activate the environment `mamba activate ./.env`
3. (Optional) deactivate the environment `mamba deactivate`
4. Run using `python src/main.py [subcommand] [arguments or config]`
    - Example using arguments: `python src/main.py fit --dataset.class_path WikitextDataset --data.head 10 --model MirrorLlamaModel --model.id 3.2-1B-Instruct --slurm.gpus_per_node h200:1`
    - Example using config: `python src/main.py fit --config config.yaml`
      - Example config.yaml contents: 
      ```
      data:
        class_path: WikitextDataset # Note that class name or full class path works
        init_args:
          split: train
          head: 10

      model:
        class_path: mirror.models.mirror_llama_model.MirrorLlamaModel
        init_args:
          initialization:
            # 3.2-1B
            3.2-1B-Instruct

            # Can use either a pretrained model (above) or customize the config (below)

            # vocab_size: 128256 # Default: 128256
            # hidden_size: 1024 # Default: 4096 
            # intermediate_size: 2048 # Default: 11008
            # num_hidden_layers: 8 # Default: 32
            # num_attention_heads: 8 # Default: 32
            # num_key_value_heads: null # Default: null

      slurm:
        submit: true
        time: "01:00:00"
        gpus_per_node: p100:1 # Use 1 P100 GPU. Options: A100, A200, P100, L40S, H200
        mem_per_cpu: "128G"
        output: "slurm_logs/%j.out"
        open_mode: "append"
        signal: "SIGHUP@90"
        requeue: true

      epochs: 1
      batch_size: 1
      device: cpu
      ```

5. Weights & Biases tracking is enabled by default through `Trainer`.
    - On SLURM compute nodes it defaults to offline mode and writes runs to `~/nobackup/autodelete/mirror_data/wandb`.
    - Sync cached runs later from a login node with `wandb sync ~/nobackup/autodelete/mirror_data/wandb/offline-run-*`.
