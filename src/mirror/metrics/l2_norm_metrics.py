from typing import Any, Mapping, cast

import torch
from lightning import Fabric

from mirror.metrics.mirror_metric import MirrorMetric
from mirror.models.mirror_model import MirrorModel


class L2NormMetrics[RawT: Mapping[str, Any], ProcessedT: Mapping[str, Any], BatchT, ModelOutputT](
    MirrorMetric[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def get_metrics(self, model: MirrorModel[RawT, ProcessedT, BatchT, ModelOutputT], fabric: Fabric) -> dict:
        params = [p.detach() for p in model.parameters()]
        if not params:
            return {"l2_norm": 0.0}
        per_param_norms = torch.stack([torch.linalg.vector_norm(p) for p in params])
        local_sq = per_param_norms.pow(2).sum()
        global_sq = cast(torch.Tensor, fabric.all_reduce(local_sq, reduce_op="sum"))
        return {"l2_norm": global_sq.sqrt().item()}
