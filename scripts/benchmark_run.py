#!/usr/bin/env python
"""Run a single benchmark configuration on a compute node.

Called by benchmark_launch.py via sbatch/srun. Trains for one epoch on
church-english.txt (100k samples) with a randomly initialised Llama model
of the specified size, using TimerCallback to log the result.
"""
import argparse
from pathlib import Path

from mirror.callbacks.timer_callback import TimerCallback
from mirror.config import init_config
from mirror.datasets.txt_dataset import TxtDataset
from mirror.models.configuration_llama import LlamaConfig
from mirror.models.mirror_llama_model import MirrorLlamaModel
from mirror.trainer import Trainer
from mirror.util import mirror_data_path


MODEL_CONFIGS: dict[str, LlamaConfig] = {
    "100k": LlamaConfig(vocab_size=6000, hidden_size=16,  intermediate_size=24,   num_hidden_layers=4,  num_attention_heads=16, tie_word_embeddings=True),
    "1M":   LlamaConfig(vocab_size=8000, hidden_size=96,  intermediate_size=128,  num_hidden_layers=4,  num_attention_heads=16, tie_word_embeddings=True),
    "10M":  LlamaConfig(vocab_size=8000, hidden_size=360, intermediate_size=400,  num_hidden_layers=8,  num_attention_heads=16, tie_word_embeddings=True),
    "100M": LlamaConfig(vocab_size=8000, hidden_size=896, intermediate_size=1000, num_hidden_layers=16, num_attention_heads=16, tie_word_embeddings=True),
}

DATASET_PATH = Path("/grphome/grp_mirror/mirror_data/datasets/church-english.txt")
DATASET_HEAD = 100_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes",  type=int, required=True)
    parser.add_argument("--devices",    type=int, required=True, help="GPUs per node")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS), required=True)
    parser.add_argument("--log-file",   type=Path, default=mirror_data_path / "benchmark_log.jsonl")
    parser.add_argument("--lock-dir",   type=Path, default=mirror_data_path / "timer_locks")
    args = parser.parse_args()

    init_config()

    model = MirrorLlamaModel(initialization=MODEL_CONFIGS[args.model_size], seed=42)
    dataset = TxtDataset(DATASET_PATH, head=DATASET_HEAD)

    timer = TimerCallback(log_file=args.log_file, lock_dir=args.lock_dir)
    trainer = Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        callbacks=[timer],
    )
    trainer.launch()
    trainer.fit(model, dataset, batch_size=args.batch_size, epochs=1)


if __name__ == "__main__":
    main()
