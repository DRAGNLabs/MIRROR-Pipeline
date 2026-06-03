from typing import cast

from transformers import BatchEncoding, PreTrainedTokenizerBase

from mirror.preprocessors.mirror_preprocessor import MirrorPreprocessor
from mirror.preprocessors.preprocessor_util import collate_tokens, load_hf_tokenizer
from mirror.types import IGNORE_ID, LabeledTokens, PromptResponseRow, StandardBatch, TokenTensor


class MirrorSftPreprocessor(
    MirrorPreprocessor[PromptResponseRow, LabeledTokens, StandardBatch]
):
    """
    Tokenizes prompt and response separately so the loss can be masked on
    prompt tokens. If the tokenizer has a chat template (most HF instruct
    models do), wraps prompt/response as a user/assistant conversation so
    the model's expected role markers and special tokens are emitted.
    Emits one input_ids sequence (prompt + response) and a matching labels
    sequence with IGNORE_ID over prompt positions.
    """

    def __init__(self, hf_model_name: str, max_length: int | None = 2048) -> None:
        self._tokenizer: PreTrainedTokenizerBase = load_hf_tokenizer(hf_model_name)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._max_length = max_length

    def preprocess_example(self, example: PromptResponseRow) -> LabeledTokens:
        prompt_ids, response_ids = self._tokenize_split(example)
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_ID] * len(prompt_ids) + response_ids

        if self._max_length is not None and len(input_ids) > self._max_length:
            input_ids = input_ids[:self._max_length]
            labels = labels[:self._max_length]

        return LabeledTokens(input_ids=input_ids, labels=labels)

    def _tokenize_split(self, example: PromptResponseRow) -> tuple[list[int], list[int]]:
        if self._tokenizer.chat_template is not None:
            full_msgs = [
                {"role": "user", "content": example['prompt']},
                {"role": "assistant", "content": example['response']},
            ]
            out = cast(BatchEncoding, self._tokenizer.apply_chat_template(
                full_msgs,
                return_dict=True,
                return_assistant_tokens_mask=True,
            ))
            full_ids = list(cast(list[int], out["input_ids"]))
            mask = cast(list[int], out["assistant_tokens_mask"])
            split = next((i for i, m in enumerate(mask) if m == 1), len(full_ids))
            return full_ids[:split], full_ids[split:]

        prompt_ids = self._tokenizer.encode(example['prompt'], add_special_tokens=True)
        response_ids = self._tokenizer.encode(example['response'], add_special_tokens=False)
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None and (len(response_ids) == 0 or response_ids[-1] != eos_id):
            response_ids = [*response_ids, eos_id]
        return cast(TokenTensor, list(prompt_ids)), cast(TokenTensor, list(response_ids))

    def collate(self, examples: list[LabeledTokens]) -> StandardBatch:
        return collate_tokens(self._tokenizer, examples)

    @property
    def pad_token_id(self) -> int:
        return cast(int, self._tokenizer.pad_token_id)
