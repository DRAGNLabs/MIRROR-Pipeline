import os
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
        self._load_checkpoint(model, fabric, checkpoint_path)
        pipe = self._build_pipeline(model, fabric, formatter)
        generation_kwargs = self._generation_kwargs(max_new_tokens, temperature, top_p, top_k, repetition_penalty)

        result = pipe(text, **generation_kwargs)
        return result[0]['generated_text']

    def predict_interactive(
            self,
            model: InferenceModel,  # type: ignore[type-arg]
            fabric: Fabric,
            max_new_tokens: int,
            checkpoint_path: str | None = None,
            formatter: InferFriendlyFormatter | None = None,
            temperature: float = 1.0,
            top_p: float | None = None,
            top_k: int | None = None,
            repetition_penalty: float = 1.0,
    ) -> None:
        from prompt_toolkit import PromptSession

        self._load_checkpoint(model, fabric, checkpoint_path)
        pipe = self._build_pipeline(model, fabric, formatter)
        generation_kwargs = self._generation_kwargs(max_new_tokens, temperature, top_p, top_k, repetition_penalty)

        session: PromptSession[str] = PromptSession(multiline=True, prompt_continuation="... ")
        try:
            while True:
                text = session.prompt("prompt> ")
                if not text.strip():
                    continue
                print(pipe(text, **generation_kwargs)[0]['generated_text'])
        except (KeyboardInterrupt, EOFError):
            print()  # trailing newline so the shell prompt is on a new line

    def _load_checkpoint(
            self,
            model: InferenceModel,  # type: ignore[type-arg]
            fabric: Fabric,
            checkpoint_path: str | None,
    ) -> None:
        if checkpoint_path is not None:
            if os.path.isdir(checkpoint_path):
                import torch.distributed.checkpoint as dcp
                model_state = model.state_dict()
                dcp.load(state_dict={'model': model_state}, checkpoint_id=str(checkpoint_path), no_dist=True)  # type: ignore[attr-defined]
                model.load_state_dict(model_state)
            else:
                fabric.load(checkpoint_path, {'model': model})

    def _build_pipeline(
            self,
            model: InferenceModel,  # type: ignore[type-arg]
            fabric: Fabric,
            formatter: InferFriendlyFormatter | None,
    ):
        model.eval()

        active_formatter = formatter or cast(InferFriendlyFormatter, model.formatter)

        return hf_pipeline(
            'text-generation',
            model=model.hf_model,
            tokenizer=active_formatter.tokenizer,
            device=fabric.device,
        )

    def _generation_kwargs(
            self,
            max_new_tokens: int,
            temperature: float,
            top_p: float | None,
            top_k: int | None,
            repetition_penalty: float,
    ) -> dict:
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
        return generation_kwargs
