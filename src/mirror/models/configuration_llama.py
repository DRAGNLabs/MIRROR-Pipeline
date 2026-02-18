from transformers import LlamaConfig as TransformersLlamaConfig


class LlamaConfig(TransformersLlamaConfig):
    def __init__(
        self,
        vocab_size: int | None = 128256,
        rms_norm_eps: float | None = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            **kwargs,
        )
