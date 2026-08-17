import os
from lightning import Fabric
from transformers import pipeline as hf_pipeline

from mirror.fabric_util import rank_zero_log
from mirror.formatters.infer_friendly_formatter import InferFriendlyFormatter
from mirror.models.inference_model import InferenceModel

class Predictor:
    def predict(
            self,
            model: InferenceModel,  # type: ignore[type-arg]
            fabric: Fabric,
            text: str,
            max_new_tokens: int,
            checkpoint_path: str | None = None,
            formatter: InferFriendlyFormatter | None = None,
            temperature: float = 0.0,
            top_p: float | None = None,
            top_k: int | None = None,
            repetition_penalty: float = 1.0,
    ) -> str:
        if checkpoint_path is not None:
            # FSDP saves sharded checkpoints as a directory; fabric.load can't restore those into
            # the unwrapped inference model, so dcp reads them in a single process. Single-file
            # checkpoints (non-FSDP runs) load normally through fabric.
            if os.path.isdir(checkpoint_path):
                import torch.distributed.checkpoint as dcp
                model_state = model.state_dict()
                dcp.load(state_dict={'model': model_state}, checkpoint_id=str(checkpoint_path), no_dist=True)  # type: ignore[attr-defined]
                model.load_state_dict(model_state)
            else:
                fabric.load(checkpoint_path, {'model': model})

        model.eval()

        active_formatter = formatter or model.formatter

        pipe = hf_pipeline(
            'text-generation',
            model=model.hf_model,
            tokenizer=active_formatter.tokenizer,
            device=fabric.device,
        )

        do_sample = temperature > 0.0 or top_p is not None or top_k is not None
        generation_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        if temperature > 0.0:  # HF requires temperature > 0 when sampling; temperature=0 is greedy
            generation_kwargs['temperature'] = temperature
        if top_p is not None:
            generation_kwargs['top_p'] = top_p
        if top_k is not None:
            generation_kwargs['top_k'] = top_k

        self._log_inference_config(fabric, checkpoint_path, generation_kwargs)
        result = pipe(text, **generation_kwargs)
        return result[0]['generated_text']

    @staticmethod
    def _log_inference_config(fabric: Fabric, checkpoint_path: str | None, generation_kwargs: dict) -> None:
        rank_zero_log(fabric, f"Inference checkpoint: {checkpoint_path or 'none (base model weights)'}")
        rank_zero_log(fabric, f"Generation config: {generation_kwargs}")
