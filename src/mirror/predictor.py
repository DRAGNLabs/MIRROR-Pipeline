from pathlib import Path

import torch
from transformers import pipeline as hf_pipeline

from mirror.models.mirror_model import MirrorModel
from mirror.models.whitebox_transformers.hf_whitebox_transformers import HFWhiteboxTransformer
from mirror.util import get_device


class Predictor:
    def predict(
            self,
            model: MirrorModel,
            checkpoint_path: Path,
            text: str,
            num_tokens: int,
    ) -> str:
        if not isinstance(model, HFWhiteboxTransformer):
            raise ValueError(
                f"prediction requires a model that implements HFWhiteboxTransformer, got {type(model)}"
            )

        device = get_device()

        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        model.eval()

        pipe = hf_pipeline(
            'text-generation',
            model=model.hf_model,
            tokenizer=model.preprocessor._tokenizer,
            device=device,
        )

        print(pipe.device)

        result = pipe(text, max_new_tokens=num_tokens, do_sample=False)
        return result[0]['generated_text']
