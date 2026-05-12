import math
from typing import cast

import torch
from lightning import Fabric

from mirror.datasets.mirror_dataset import MirrorDataset
from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.mirror_model import MirrorModel
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.types import AttentionMaskBatch, TextRow, TokenBatch, TokenTensor


class BitsPerByteMetric[ModelOutputT](
    MirrorMetric[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], ModelOutputT]
):
    def __init__(
            self,
            data: MirrorDataset,
            preprocessor: MirrorPreprocessor | None = None,
    ) -> None:
        self.data = data
        self.preprocessor = preprocessor

    def get_metrics(
            self,
            model: MirrorModel[TextRow, TokenTensor, tuple[TokenBatch, AttentionMaskBatch], ModelOutputT],
            fabric: Fabric,
    ) -> dict:
        preprocessor = self.preprocessor or model.preprocessor
        local_indices = range(fabric.global_rank, len(self.data), fabric.world_size)

        total_bits = 0.0
        total_bytes = 0

        with torch.no_grad():
            for i in local_indices:
                row: TextRow = self.data[i]
                tokens = preprocessor.preprocess_example(row)
                batch = preprocessor.collate([tokens])

                num_tokens = len(tokens)
                num_bytes = len(row['text'].encode('utf-8'))

                loss_nats = model.training_step(batch).loss.item()

                # causal LM predicts T-1 targets from T tokens
                total_bits += loss_nats / math.log(2) * (num_tokens - 1)
                total_bytes += num_bytes

        total_bits_global = cast(torch.Tensor, fabric.all_reduce(torch.tensor(total_bits, device=fabric.device), reduce_op="sum")).item()
        total_bytes_global = cast(torch.Tensor, fabric.all_reduce(torch.tensor(total_bytes, device=fabric.device), reduce_op="sum")).item()

        return {"bits_per_byte": total_bits_global / total_bytes_global}
