from typing import cast
from lightning import Fabric
from transformers import pipeline as hf_pipeline

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
            temperature: float = 1.0,
            top_p: float | None = None,
            top_k: int | None = None,
            repetition_penalty: float = 1.0,
    ) -> str:
        if checkpoint_path is not None:
            fabric.load(checkpoint_path, {'model': model})

        model.eval()

        active_formatter = formatter or cast(InferFriendlyFormatter, model.formatter)

        pipe = hf_pipeline(
            'text-generation',
            model=model.hf_model,
            tokenizer=active_formatter.tokenizer,
            device=fabric.device,
        )

        do_sample = temperature != 1.0 or top_p is not None or top_k is not None
        generation_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        if top_p is not None:
            generation_kwargs['top_p'] = top_p
        if top_k is not None:
            generation_kwargs['top_k'] = top_k

        result = pipe(text, **generation_kwargs)
        return result[0]['generated_text']
