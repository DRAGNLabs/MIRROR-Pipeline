End-to-end pipeline built atop megatron. Also built atop [[Lightning]].

> All NeMo models are trained with [Lightning](https://github.com/Lightning-AI/lightning).

[github](https://github.com/NVIDIA-NeMo/NeMo?tab=readme-ov-file)

Broad support for HuggingFace models. 


>When applicable, NeMo models leverage cutting-edge distributed training techniques, incorporating [parallelism strategies](https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html) to enable efficient training of very large models. These techniques include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed Precision Training with BFloat16 and FP8, as well as others.

> NeMo Transformer-based LLMs and MMs utilize [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on NVIDIA Hopper GPUs, while leveraging [NVIDIA Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for scaling Transformer model training.

> NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF). See [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

> In addition to supervised fine-tuning (SFT), NeMo also supports the latest parameter efficient fine-tuning (PEFT) techniques such as LoRA, P-Tuning, Adapters, and IA3. Refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html) for the full list of supported models and techniques.