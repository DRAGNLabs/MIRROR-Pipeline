Transformer Engine (from NVIDIA) is a library that adds:

- optimized Transformer layers (attention, feedforward)
    
- automatic mixed-precision (FP8/FP16/BF16)
    
- dynamic scaling and quantization
    
- specialized kernels for H100 (Hopper) and later GPUs
    

It’s deeply integrated into **Megatron-Core** and **NVIDIA NeMo**, and provides:

- cuDNN FlashAttention kernels (on Hopper)
    
- fallback to Tri Dao’s FlashAttention kernels (on Ampere)
    
- fused layernorm, softmax, and linear ops