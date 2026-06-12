from typing import Any, Mapping, cast

import torch
from lightning import Fabric

from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.mirror_model import MirrorModel


class GradNormMetrics[RawT: Mapping[str, Any], FormattedT: Mapping[str, Any], BatchT, ModelOutputT](
    MirrorMetric[RawT, FormattedT, BatchT, ModelOutputT]
):
    def get_metrics(self, model: MirrorModel[RawT, FormattedT, BatchT, ModelOutputT], fabric: Fabric) -> dict:
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if not grads:
            return {"grad_norm": 0.0}
        per_param_norms = torch.stack([torch.linalg.vector_norm(g) for g in grads])
        local_sq = per_param_norms.pow(2).sum()
        global_sq = cast(torch.Tensor, fabric.all_reduce(local_sq, reduce_op="sum"))
        return {"grad_norm": global_sq.sqrt().item()}
