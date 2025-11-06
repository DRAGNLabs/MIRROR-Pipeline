https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

originally proposed in [[Megatron-LM]] paper. 

> Tensor Parallel (TP) was originally proposed in the [Megatron-LM](https://arxiv.org/abs/1909.08053) paper, and it is an efficient model parallelism technique to train large scale Transformer models. [Sequence Parallel](https://arxiv.org/abs/2205.05198) (SP) we mention in this tutorial is a variant of Tensor Parallel that shards on the sequence dimension for `nn.LayerNorm` or `RMSNorm` to further save activation memory during training. As the model becomes larger, the activation memory becomes the bottleneck, so in Tensor Parallel training it usually applies Sequence Parallel to `LayerNorm` or `RMSNorm` layers.

Apr 19, 2024, updated Jul 18, 2025.

Experimental in Pytorch. Ofc Megatron's version is the og. 
# what it is
from [HF](https://huggingface.co/docs/transformers/v4.57.0/perf_train_gpu_many#tensor-parallelism)

> Tensor parallelism distributes large tensor computations across multiple GPUs. The tensors are sliced horizontally or vertically and each slice is processed by a separate GPU. Each GPU performs its calculations on its tensor slice and the results are synchronized at the end to reconstruct the final result.

>Tensor parallelism is effective for training large models that don’t fit into the memory of a single GPU. It is also faster and more efficient because each GPU can process its tensor slice in parallel, and it can be combined with other parallelism methods. Like other parallelism methods though, tensor parallelism adds communication overhead between GPUs.


## When and Why you should apply Tensor Parallel
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html#when-and-why-you-should-apply-tensor-parallel
The PyTorch Fully Sharded Data Parallel (FSDP) already has the capability to scale model training to a specific number of GPUs. However, when it comes to further scale the model training in terms of model size and GPU quantity, many additional challenges arise that may require combining Tensor Parallel with FSDP.:

1. As the world size (number of GPUs) is becoming excessively large (exceeding 128/256 GPUs), the FSDP collectives (such as `allgather`) are being dominated by ring latency. By implementing TP/SP on top of FSDP, the FSDP world size could be reduced by 8 by applying FSDP to be inter-host only, consequently decreasing the latency costs by the same amount.
    
2. Hit data parallelism limit where you can not raise the global batch size to be above the number of GPUs due to both convergence and GPU memory limitations, Tensor/Sequence Parallel is the only known way to “ballpark” the global batch size and continue scaling with more GPUs. This means both model size and number of GPUs could continue to scale.
    
3. For certain types of models, when local batch size becomes smaller, TP/SP can yield matrix multiplication shapes that are more optimized for floating point operations (FLOPS).
    

So, when pre-training, how easy is it to hit those limits? As of now, pre-training a Large Language Model (LLM) with billions or trillions of tokens could take months, even when using thousands of GPUs.

- It will always hit limitation 1 when training LLM on a large scale. For example, Llama 2 70B trained with 2k GPUs for 35 days, multi-dimensional parallelisms are needed at 2k scale.
    
- When the Transformer model becomes larger (such as Llama2 70B), it will also quickly hit the limitation 2. One could not use FSDP alone with even local `batch_size=1` due to memory and convergence constraints. For example, Llama 2 global batch size is 1K, so data parallelism alone can not be used at 2K GPUs.