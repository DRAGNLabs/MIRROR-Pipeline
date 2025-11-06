

[archiv](https://arxiv.org/abs/2205.14135)
[github](https://github.com/Dao-AILab/flash-attention)

Supports cuda, as well as rocm, [[Triton]] backend currently in progress. 

> IO-aware -- accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.

designed for **both training and inference**. It’s a drop-in, _exact_ (up to fp precision) attention kernel that speeds up forward **and backward** passes while cutting memory use, so you can train with longer sequences and larger batches. ([PyTorch](https://pytorch.org/blog/out-of-the-box-acceleration/?utm_source=chatgpt.com "Out of the box acceleration and memory savings of 🤗 ..."))

# what it's used in
- **PyTorch:** yes—built in via SDPA. You can opt into the **FLASH_ATTENTION** backend (or let PyTorch auto-select it) with the `sdp_kernel` context manager. Works for forward **and backward** in training. [PyTorch Documentation+2PyTorch Documentation+2](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html?utm_source=chatgpt.com)
    
- **Hugging Face Transformers:** they don’t “own” a separate FA kernel; they expose an **attention interface** that can use:
    
    - PyTorch SDPA (`attn_implementation="sdpa"`), or
        
    - **flash-attn** (Tri Dao’s package) if you install it (`attn_implementation="flash_attention_2"`). [Hugging Face+1](https://huggingface.co/docs/transformers/en/attention_interface?utm_source=chatgpt.com)
        
- **Megatron(-Core)/NeMo:** supports flash-style attention via **Transformer Engine** (selects Tri Dao FA vs cuDNN flash attention depending on GPU; Hopper prefers cuDNN, Ampere prefers Tri Dao FA). Megatron-Core is the training stack NeMo builds on. [NVIDIA Docs+1](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/attention_optimizations.html?utm_source=chatgpt.com)
    
- **DeepSpeed:** has its own fused **Transformer Kernel** (separate from Tri Dao’s package) and is commonly used alongside other scale techniques (ZeRO, Ulysses, etc.). You can train with DeepSpeed and either rely on its kernel or integrate flash-attn in your model code. [DeepSpeed+2DeepSpeed+2](https://www.deepspeed.ai/tutorials/transformer_kernel/?utm_source=chatgpt.com)
    
- **MosaicML Composer:** supports FA through the underlying model implementations. Example: Mosaic’s MPT uses a Triton flash-attention kernel; Composer also wraps HF models, so the HF `attn_implementation` path works there too. [Hugging Face](https://huggingface.co/mosaicml/mpt-7b-instruct/blob/main/flash_attn_triton.py?utm_source=chatgpt.com)
# use it in examples
**PyTorch (2.x) – SDPA “flash” backend**

- Call `torch.nn.functional.scaled_dot_product_attention` (used under the hood by many modules). On supported GPUs, PyTorch will pick a fused FlashAttention/SDPA kernel automatically. You can force/select kernels via the backend toggles:
    
    ```python
    import torch
    from torch.backends.cuda import sdp_kernel
    # Prefer flash; fall back as needed
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    ```
    
    This works in **training** (dropout, masks, and autograd are supported). Requirements: recent CUDA + NVIDIA Ampere/Hopper–class GPUs (SM80+). ([PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention"))
    

**Hugging Face Transformers**

- Many decoder models allow:
    
    ```python
    model = AutoModelForCausalLM.from_pretrained(
        "your/model",
        attn_implementation="flash_attention_2"   # if flash-attn is installed
    )
    ```
    
    Install the kernels (matching your CUDA & GPU): `pip install flash-attn` (or build from source for the newest FA3). FA2/FA3 report end-to-end **training** speedups over baseline attention. ([GitHub](https://github.com/Dao-AILab/flash-attention?utm_source=chatgpt.com "Dao-AILab/flash-attention: Fast and memory-efficient ..."))
    



Reported end-to-end training speedups (GPT-style models) and high FLOP utilization on A100/H100; FA3 further boosts Hopper (H100) with warp-specialization and FP8 paths. ([Princeton NLP](https://princeton-nlp.github.io/flash-atttention-2/?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism and ..."))