from transformers import PreTrainedTokenizerBase

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import collate_tokens, load_hf_tokenizer
from mirror.types import AttentionMaskBatch, IGNORE_ID, LabeledTokens, LabelsBatch, PromptResponseRow, TokenBatch


class MirrorSftPreprocessor(
    MirrorPreprocessor[PromptResponseRow, LabeledTokens, tuple[TokenBatch, AttentionMaskBatch, LabelsBatch]]
):
    """
    Tokenizes prompt and response separately so the loss can be masked on
    prompt tokens. Emits one input_ids sequence (prompt + response) and a
    matching labels sequence with IGNORE_ID over prompt positions.
    """

    def __init__(self, hf_model_name: str, max_length: int | None = 2048) -> None:
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._max_length = max_length

    def preprocess_example(self, example: PromptResponseRow) -> LabeledTokens:
        prompt_ids = self._tokenizer.encode(example['prompt'], add_special_tokens=True)
        response_ids = self._tokenizer.encode(example['response'], add_special_tokens=False)
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None and (len(response_ids) == 0 or response_ids[-1] != eos_id):
            response_ids = [*response_ids, eos_id]

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_ID] * len(prompt_ids) + response_ids

        if self._max_length is not None and len(input_ids) > self._max_length:
            input_ids = input_ids[:self._max_length]
            labels = labels[:self._max_length]

        return LabeledTokens(input_ids=input_ids, labels=labels)

    def collate(self, examples: list[LabeledTokens]) -> tuple[TokenBatch, AttentionMaskBatch, LabelsBatch]:
        return collate_tokens(self._tokenizer, examples)

    @property
    def pad_token_id(self) -> int:
        return int(self._tokenizer.pad_token_id)  # type: ignore[arg-type]
