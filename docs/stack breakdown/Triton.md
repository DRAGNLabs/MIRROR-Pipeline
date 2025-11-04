**Triton** is an **open-source GPU programming framework** developed by **OpenAI** (and now part of the PyTorch ecosystem) that makes it easier to write **high-performance custom GPU kernels** — similar in spirit to CUDA, but much higher level and more Python-friendly.

Here’s the breakdown:

---

### 🧠 **What Triton Is**

- A **domain-specific language (DSL)** and compiler embedded in Python.
    
- Lets you write GPU kernels in Python using a syntax close to NumPy or PyTorch, which Triton then compiles to efficient GPU code.
    
- Designed to make it easy for ML researchers to optimize performance-critical parts of models (like attention, matmul, layernorm) without needing to dive deep into CUDA C++.
    

So instead of writing thousands of lines of CUDA to fuse kernels, you can write a few dozen lines of Triton.

---

### ⚙️ **Why It Matters for Flash Attention**

FlashAttention v2 is all about **memory-efficient and bandwidth-optimized attention kernels**. Implementations exist in CUDA, but:

- **Triton** makes it easier to write and port such kernels to **different GPU architectures**, like AMD’s CDNA and RDNA.
    
- It enables **autotuning**, **data-type specialization (fp16, bf16, fp8)**, and **efficient tiling** with far less boilerplate.
    
- Since it integrates well with **PyTorch**, a Triton backend can slot directly into existing PyTorch ops.
    

That’s why you see the note saying:

> “The Triton implementation of the Flash Attention v2 is currently a work in progress.”

They’re rewriting the kernel in Triton to make it:

- Easier to maintain,
    
- More portable (esp. to AMD GPUs),
    
- Potentially faster via better tile optimization.
    

---

### 🚀 **Where It Fits**

|Framework|Purpose|
|---|---|
|**CUDA / HIP**|Low-level GPU APIs for Nvidia / AMD|
|**Triton**|Higher-level DSL that compiles to those backends|
|**PyTorch**|ML framework that can call Triton kernels|
|**FlashAttention (Triton backend)**|Specific optimized attention op implemented in Triton|

---

### 🧩 **Key Features Triton Enables**

- Custom GPU kernels written in Python.
    
- Automatic memory coalescing and parallelization.
    
- Support for fp16, bf16, fp8, etc.
    
- Performance comparable to hand-tuned CUDA in many cases.
    
- Works on both Nvidia and AMD GPUs (via ROCm).
    

---

From the [[FlashAttention]] github:
Phil Tillet (OpenAI) has an experimental implementation of FlashAttention in Triton: [https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py](https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py)

As Triton is a higher-level language than CUDA, it might be easier to understand and experiment with. The notations in the Triton implementation are also closer to what's used in our paper.

We also have an experimental implementation in Triton that support attention bias (e.g. ALiBi): [https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)