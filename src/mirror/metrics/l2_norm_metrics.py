from typing import cast

import torch
from lightning import Fabric

from mirror.metrics.extra_metrics_getter import ExtraMetricsGetter
from mirror.models.trainable_model import TrainableModel


class L2NormMetrics(ExtraMetricsGetter):
    def get_metrics(self, model: TrainableModel, fabric: Fabric) -> dict:
        params = [p.detach() for p in model.parameters()]
        if not params:
            return {"l2_norm": 0.0}
        per_param_norms = torch.stack([torch.linalg.vector_norm(p) for p in params])
        local_sq = per_param_norms.pow(2).sum()
        global_sq = cast(torch.Tensor, fabric.all_reduce(local_sq, reduce_op="sum"))
        return {"l2_norm": global_sq.sqrt().item()}
