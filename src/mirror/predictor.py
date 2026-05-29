import torch.distributed.checkpoint as dcp
from typing import cast
from transformers import pipeline as hf_pipeline, PreTrainedTokenizerFast

from mirror.models.inference_model import InferenceFriendlyModel
from mirror.preprocessors.mirror_preprocessor import InferenceFriendlyPreprocessor
from mirror.util import get_device

class Predictor:
    def predict(
            self,
            model: InferenceFriendlyModel,
            text: str,
            num_tokens: int,
            checkpoint_path: str | None = None,
            preprocessor: InferenceFriendlyPreprocessor | None = None,
            temperature: float = 1.0,
            top_p: float | None = None,
            top_k: int | None = None,
            repetition_penalty: float = 1.0,
    ) -> str:
        device = get_device()

        if checkpoint_path is not None:
            model_state = model.state_dict()
            dcp.load(  # type: ignore[attr-defined]
                state_dict={'model': model_state},
                checkpoint_id=str(checkpoint_path),
                no_dist=True,
            )
            model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        tokenizer = preprocessor.tokenizer if preprocessor is not None else model.preprocessor.tokenizer

        pipe = hf_pipeline(
            'text-generation',
            model=model.hf_model,
            tokenizer=cast(PreTrainedTokenizerFast, tokenizer),
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
