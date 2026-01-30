import os

from huggingface_hub import login

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from mirror.models.mirror_llama_model import MirrorLlamaModel


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(".ENV")

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)

    model = MirrorLlamaModel()
    print("loaded:", type(model.model).__name__)
    print("pad_token_id:", model.tokenizer.pad_token_id)


if __name__ == "__main__":
    main()
