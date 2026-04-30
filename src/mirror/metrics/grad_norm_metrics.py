from typing import cast

import torch
from lightning import Fabric

from mirror.metrics.extra_metrics_getter import ExtraMetricsGetter
from mirror.models.mirror_model import MirrorModel


class GradNormMetrics(ExtraMetricsGetter):
    def get_metrics(self, model: MirrorModel, fabric: Fabric) -> dict:
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if not grads:
            return {"grad_norm": 0.0}
        per_param_norms = torch.stack([torch.linalg.vector_norm(g) for g in grads])
        local_sq = per_param_norms.pow(2).sum()
        global_sq = cast(torch.Tensor, fabric.all_reduce(local_sq, reduce_op="sum"))
        return {"grad_norm": global_sq.sqrt().item()}
