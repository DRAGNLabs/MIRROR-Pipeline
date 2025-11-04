
# Jaden's Recommendation:
Lightning [Fabric](https://github.com/Lightning-AI/pytorch-lightning?tab=readme-ov-file#lightning-fabric-expert-control), from the start. We would like to have accelerators, distributed strategies, logging/checkpointing handled for us, but modules and trainers of our own. 
![[Pasted image 20251013171606.png|1200]]
Lightning Fabric is what's under the hood of NeMo, an end-to-end very large LLM model trainer actively maintained by NVIDIA. That's a big endorsement, demonstrating that it's optimal and imitatably in use for the scale that we're looking to reach. 


[[big stack breakdowns]]

# must-have tech

[[FSDP]], framework for parallelization of components related to the backward pass. Builds off DDP (Distributed Data Parallel), which runs the full model on different batchProvide fused CUDA kernels or its own tensor-parallel scheme — it plugs into others (like Megatron).es, limiting model size to single GPU VRAM. Originally part of [[FairScale]], made by Facebook.

[[Tensor Parallel]], originally part of [[Megatron-LM]], is made for deeper splitting, within the layer itself. 

[[Megatron-LM|Megatron]] vs [[DeepSpeed]]: 
- Once used together. 
- DeepSpeed: Memory sharding of optimizer states (ZeRO), CPU/NVMe offload, cluster orchestration.
- Megatron: Provide fused CUDA kernels or its own tensor-parallel scheme — it plugs into others (like Deepspeed).
- sounds like both may have since grown to encompass the others.

[[NeMo]]: end-to-end ready pipeline built atop Megatron. 

[[FlashAttention]] is an optimization on IO btw levels of GPU memory. Built into pytorch via [[SDPA]], HF has it via pytorch, the og flashattention github (from Tri Dao, author of the paper). implementation, and `eager` (just a matmul). Mosaic has it via custom [[Triton]] kernel, but can also wrap the HF ones. Megatron/NeMo does it via [[Transformer Engine]], an Nvidia library for lots of stuff, one of them being custom flashattention kernels that fall back on Dao's og. So pretty much only DeepSpeed does its own thing. 

**Composer** vs **Lightning**: They're both equivalent to each other in abstraction, both beefier versions of HF **Accelerate**. Orchestrators. Very flexible to underlying algorithms used. Composer isn't really great at some of the super deep features. 

**Accelerate** has the advantage of being simple, but isn't quite built for big scale. However, both [Megatron-core](https://github.com/NVIDIA-NeMo/Megatron-Bridge?tab=readme-ov-file) and [DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) integrate with it. 

## 3d parallelism
[[Nanotron]]: Minimalistic large language model 3D-parallelism training. Primarily Data parallel, tensor parallel, and pipeline parallel. 

Data Parallel: [[FSDP]], DeepSpeed ZeRO (3rd part)
[[Tensor Parallel]]: Shards on sequence dimension for `LayerNorm` or `RMSNorm`. The tensors are sliced horizontally or vertically and each slice is processed by a separate GPU.
[[Pipeline Parallelism]]: micro-batches the data.

Only Nanotron, Megatron, and DeepSpeed seem to be able to do all 3, out of the box. Strangely, Lightning only has an experimental implementation, so NeMo appears to have a special [bridge](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemo-2.0/features/megatron.html) (more likely the other way around, tbh). 
What techniques do I need? Use the [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage). Then use this [table](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) from HF:

|setup|scenario|strategy|
|---|---|---|
|single node/multi-GPU|fits on single GPU|DistributedDataParallel or ZeRO|
||doesn’t fit on single GPU|PipelineParallel, ZeRO or TensorParallel|
||largest model layer doesn’t fit|TensorParallel or ZeRO|
|multi-node/multi-GPU|fast inter-node connectivity (NVLink or NVSwitch)|ZeRO or 3D parallelism (PipelineParallel, TensorParallel, DataParallel)|
||slow inter-node connectivity|ZeRO or 3D parallelism (PipelineParallel, TensorParallel, DataParallel)|

## breakdown of features

### Comprehensive Comparison Table
| **Feature**                   | **Nanotron**                  | **PyTorch Native**                                      | **Pure PyTorch Lightning**                                              | **Lightning Fabric**                        | **NeMo / Megatron**                              | **Accelerate**                                  | **Composer**                                              | **DeepSpeed**                                      |
| ----------------------------- | ----------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------- |
| **3D Parallelism (DP+TP+PP)** | ✅ All 3                       | 🟡 TP & PP available but **experimental/APIs evolving** | ⚠️ DP out-of-box; PP/TP via strategy/plugins (e.g., DeepSpeed/Megatron) | ⚠️ DP out-of-box; PP via DS; TP via plugins | ✅ All 3                                          | ⚠️ DP; PP/TP only via DeepSpeed config          | 🟡 **DP + TP** (no PP); FSDP                              | ✅ **DP + PP** natively; **TP via Megatron plugin** |
| **Expert Parallelism (MoE)**  | ✅ Yes                         | ❌ No (custom only)                                      | ⚠️ Custom/integration                                                   | ⚠️ Custom/integration                       | ✅ **Expert Parallelism** (MoE)                   | ❌ No (unless custom)                            | ❌ No                                                      | ✅ **Industry-leading** (Megatron-DS)               |
| **Pipeline Schedules**        | ✅ **1F1B, AFAB**              | 🟡 Pipeline module inc. **interleaved 1F1B**            | ⚠️ Custom only (via DS/etc.)                                            | ⚠️ Custom only (via DS/etc.)                | ✅ **All schedules incl. interleaved 1F1B / VPP** | ❌ No native pipeline                            | ❌ No pipeline                                             | ✅ **1F1B** (no interleaved 1F1B)                   |
| **ZeRO-1**                    | ✅ Custom                      | 🟡 via FSDP config                                      | 🟡 via strategy (DS/FSDP)                                               | 🟡 via strategy (DS/FSDP)                   | ✅ via backend                                    | ✅ via backend                                   | ✅ via backend                                             | ✅ Native                                           |
| **ZeRO-3 (FSDP)**             | 🔴 Not primary (ZeRO-1 focus) | ✅ **Native FSDP** (ZeRO-3-like)                         | ✅ **FSDPStrategy**                                                      | ✅ **FSDPStrategy**                          | ✅ FSDP or ZeRO-3                                 | ✅ Primary choices: FSDP or DS                   | ✅ **FSDP** (DS support now deprecated in latest releases) | ✅ **Native ZeRO-3**                                |
| **FP32 Grad Accum.**          | ✅ Built-in                    | ✅ Manual/native                                         | ✅ Built-in                                                              | ✅ Manual                                    | ✅ Supported                                      | ✅ Built-in                                      | ✅ Built-in                                                | ✅ Built-in                                         |
| **Parameter Tying/Sharding**  | ✅ Supported                   | 🟡 Possible                                             | ✅ via strategy/wrap                                                     | ✅ low-level control                         | ✅ Excellent                                      | 🟡 Possible                                     | 🟡 Possible                                               | ✅ Supported                                        |
| **Activation Checkpointing**  | ✅ Custom                      | ✅ Native                                                | ✅ Callback/config                                                       | ✅ Manual or auto                            | ✅ Optimized                                      | ✅ Via backend                                   | ✅ Via backend                                             | ✅ Selective                                        |
| **FP8 Training**              | ⚪ **Not documented**          | 🟡 Primitives/3rd-party                                 | ⚠️ Manual/integration                                                   | ⚠️ Manual/integration                       | ✅ **Transformer Engine FP8**                     | 🟡 Via integrations (TE/MS-AMP on supported hw) | ⚠️ Manual/integration                                     | 🟡 **Experimental / via integrations**             |
| **`torch.compile` Support**   | 🔴 Not a focus                | ✅ Native                                                | ✅ Native (PL ≥2.x)                                                      | ✅ Full control                              | 🟡 Partial (depends on kernels)                  | ✅ Good (plugin)                                 | ✅ Good                                                    | ⚠️ Limited (comm ops cause graph breaks)           |
| **Ring Attention**            | 🔴 Roadmap/research           | ❌ No                                                    | ⚠️ Research only                                                        | ⚠️ Research only                            | ❌ No                                             | ❌ No                                            | ❌ No                                                      | ❌ No                                               |
| **Interleaved 1F1B**          | 🔴 Roadmap                    | ✅ **Yes** (pipeline module)                             | ⚠️ Custom only                                                          | ⚠️ Custom only                              | ✅ **Yes** (VPP / interleaved)                    | ❌ No                                            | ❌ No                                                      | 🟡 **No interleaved; standard 1F1B**               |
| **Code Clarity**              | ✅✅ Excellent                  | ✅ Excellent                                             | ✅✅ Very high                                                            | ✅ Explicit/concise                          | ⚠️ Complex                                       | ✅ Simple                                        | ✅ Clear                                                   | ⚠️ Complex                                         |
| **Production Maturity**       | 🟡 Research                   | ✅ Core                                                  | ✅ Mature                                                                | ✅ Stable/lightweight                        | ✅✅ Battle-tested                                 | ✅ Mature                                        | ✅ Mature                                                  | ✅✅ Battle-tested                                   |


### Some further explanations:

#### FP32 Gradient Accumulation

Accumulate gradients over a microbatch in full precision. For when the full batch size is too big to fit in memory. Safer for convergence in mixed-precision training. 

#### Parameter Tying / Sharding
Tying: multiple components sharing the same tensor. Reduces parameter count. 

Sharding: Splits large tensors across multiple devices. we already know the three ways. 

#### Activation Checkpointing
Only a few activations saved during forward pass, the rest are recomputed during backward pass. Saves a ton of memory. 

#### FP8 Training
8-bit floating point. Reduces bandwidth and memory use. Usually mixed-precision is actually used:
- FP8 for matmuls/convolutions
- FP16/BF16 or FP32 for accumulation and normalization
**Used in:**  
NVIDIA’s **Transformer Engine**, DeepSpeed-MoE (experimental), and partially in PyTorch AMP since v2.3.

#### `torch.compile` Support

**What it is:**  
`torch.compile()` is PyTorch’s graph capture and ahead-of-time (AOT) compiler introduced in 2.0.  
It replaces the old TorchScript/JIT path with a simpler, more dynamic optimization pipeline.

**Mechanism:**

- It wraps your model, intercepts ops, and builds an intermediate graph.
- Then passes it through an optimization backend (e.g., **Inductor**, **nvFuser**, **XLA**, or **Hidet**).
- Subsequent calls run optimized kernels, often fused or re-ordered for efficiency.

**Modes:**

- `mode="default"` — safe optimizations.
- `mode="reduce-overhead"` — smaller graphs, faster launch.
- `mode="max-autotune"` — searches for best kernel configs (slower compile, faster runtime).

**Benefits:**

- 10–40% faster training or inference in many workloads.
- Kernel fusion reduces Python overhead and GPU launch latency.
- Works with AMP, DDP, FSDP, and partially with DeepSpeed/Fabric/Lightning.

**Caveats:**

- Dynamic shapes, custom ops, and communication ops (like in FSDP/DS) can break graph capture.
- That’s why large distributed systems often list “partial” or “limited” support.

#### Ring Attention

**What it is:**  
A **memory-efficient attention algorithm** designed to scale attention computation across multiple devices without full activation replication.

**Conceptually:**

- Instead of storing all Q, K, and V tensors on each GPU, **Ring Attention** partitions them across devices in a ring topology.
- Each GPU computes partial attention results and passes intermediate results to its neighbor.
- After one “ring pass,” all devices have complete attention outputs.

**Why it matters:**

- Reduces memory and communication cost from O(N²) → O(N × K) per GPU.
- Enables training very long sequence lengths (e.g., 64K–256K tokens) on multi-GPU systems.

**Status:**

- Still **research-stage**. Implementations exist in FlashAttention variants and papers from NVIDIA/Meta (e.g., “Ring Attention with Flash Decomposition”).
- Not yet integrated into core PyTorch or Lightning.

#### Interleaved 1F1B (Pipeline Scheduling)

**What it is:**  
A pipeline-parallel **scheduling strategy** for large models.  
“1F1B” = _One Forward, One Backward per microbatch._

**Vanilla 1F1B:**

- Each pipeline stage runs one forward, then one backward, in lockstep.
- Improves utilization over simple “all forward, then all backward” but still has pipeline bubbles.

**Interleaved 1F1B:**

- Splits each pipeline stage into smaller chunks (“virtual pipeline stages”).
- While chunk A of microbatch 1 is doing backward on stage 2, chunk B of microbatch 2 can already do forward on stage 1.
- Further reduces idle GPU time, increasing throughput.

**Where used:**

- Megatron-LM and NeMo’s “Virtual Pipeline Parallelism (VPP)” implement it natively.
- PyTorch pipeline parallel module supports it experimentally.
- DeepSpeed currently only supports standard 1F1B, not interleaved.

**Trade-off:**

- More communication overhead, more complex scheduling logic.
- But significant GPU utilization gains (especially for 8+ stages).
- 
based on megatron-lm paper on [archiv](https://arxiv.org/pdf/2104.04473), dated Aug. 2021
![[Pasted image 20251013112812.png]]

#### NVME/CPU offloading

Doable on lightning, but just uses deepspeed's stuff. Experimental. NeMo has CPU offloading. Needs deepspeed to do any NVMe offloading at all. 

# optimizations to focus on

- flash attention
- activation checkpointing
- BF16 mixed precision
- gradient accumulation
- torch.compile
