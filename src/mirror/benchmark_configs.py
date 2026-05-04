from mirror.models.configuration_llama import LlamaConfig

MODEL_CONFIGS: dict[str, LlamaConfig] = {
    "100k": LlamaConfig(vocab_size=6000, hidden_size=16,  intermediate_size=24,   num_hidden_layers=4,  num_attention_heads=16, tie_word_embeddings=True),
    "1M":   LlamaConfig(vocab_size=8000, hidden_size=96,  intermediate_size=128,  num_hidden_layers=4,  num_attention_heads=16, tie_word_embeddings=True),
    "10M":  LlamaConfig(vocab_size=8000, hidden_size=360, intermediate_size=400,  num_hidden_layers=8,  num_attention_heads=16, tie_word_embeddings=True),
    "100M": LlamaConfig(vocab_size=8000, hidden_size=896, intermediate_size=1000, num_hidden_layers=16, num_attention_heads=16, tie_word_embeddings=True),
}
