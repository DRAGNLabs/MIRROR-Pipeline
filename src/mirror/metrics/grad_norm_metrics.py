from mirror.metrics.extra_metrics_getter import ExtraMetricsGetter
from mirror.models.mirror_model import MirrorModel


class GradNormMetrics(ExtraMetricsGetter):
    def get_metrics(self, model: MirrorModel) -> dict:
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().pow(2).sum().item()
        return {"grad_norm": total_sq ** 0.5}
