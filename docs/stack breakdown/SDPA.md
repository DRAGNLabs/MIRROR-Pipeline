_Scaled Dot Product Attention_ 

**Definition:**  
SDPA is the mathematical operation underlying every Transformer attention block:

$$
\text{Attention}(Q, K, V) = \text{softmax}!\left(\frac{QK^T}{\sqrt{d_k}}\right)V  
$$

PyTorch 2.0 introduced `torch.nn.functional.scaled_dot_product_attention` (SDPA) as a _single fused operator_ that can automatically dispatch to different backends depending on what your GPU supports.

**Available SDPA backends:**

- **Math:** pure PyTorch implementation (slowest, always works)
- **Mem-Efficient:** fused CUDA kernel (mid-speed, moderate memory)
- **FlashAttention:** highly optimized CUDA kernel (fastest, least memory)

You can toggle them like this:

```python
from torch.backends.cuda import sdp_kernel

# Prefer flash attention if available
sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
```

So — yes, **PyTorch has FlashAttention built in** via SDPA.  
When you call `scaled_dot_product_attention`, PyTorch will silently choose the FlashAttention backend if your GPU supports it (A100, H100, etc.).