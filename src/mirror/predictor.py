from pathlib import Path

import torch.distributed.checkpoint as dcp
from transformers import pipeline as hf_pipeline

from mirror.models.mirror_model import MirrorModel
from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.util import get_device


class Predictor:
    def predict(
            self,
            model: MirrorModel,
            checkpoint_path: Path,
            text: str,
            num_tokens: int,
            preprocessor: MirrorPreprocessor | None = None,
            temperature: float = 1.0,
            top_p: float | None = None,
            top_k: int | None = None,
            repetition_penalty: float = 1.0,
    ) -> str:
        if not isinstance(model, HFWhiteboxTransformer):
            raise ValueError(
                f"prediction requires a model that implements HFWhiteboxTransformer, got {type(model)}"
            )

        device = get_device()

        model_state = model.state_dict()
        dcp.load(
            state_dict={'model': model_state},
            checkpoint_id=str(checkpoint_path),
            no_dist=True,
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        tokenizer = preprocessor._tokenizer if preprocessor is not None else model.preprocessor._tokenizer  # type: ignore[union-attr]

        pipe = hf_pipeline(
            'text-generation',
            model=model.hf_model,
            tokenizer=tokenizer,
            device=device,
        )

        do_sample = temperature != 1.0 or top_p is not None or top_k is not None
        generation_kwargs: dict = dict(
            max_new_tokens=num_tokens,
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
