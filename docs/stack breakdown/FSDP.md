Fully Sharded Data Parallel. 
[pytorch docs](https://docs.pytorch.org/docs/stable/fsdp.html)

> Using FSDP involves wrapping your module and then initializing your optimizer after. This is required since FSDP changes the parameter variables.

based on [xu et al](https://arxiv.org/abs/2004.13336) (google) and ZeRO stage 3 from [[DeepSpeed]]. Splits up weights, gradients, and optimizer states onto separate GPUs. Even Data Parallel (DP) or Distributed Data Parallel (DDP) replicates the full model on every GPU — which **wastes memory**. Not used for inference. 

> In data-parallel synchronous training of deep neural networks, different devices (replicas) run the same program with different partitions of the training batch, but weight update computation is repeated on all replicas, because the weights do not have a batch dimension to partition. This can be a bottleneck for performance and scalability in typical language models with large weights, and models with small per-replica batch size which is typical in large-scale training. This paper presents an approach to automatically shard the weight update computation across replicas with efficient communication primitives and data formatting, using static analysis and transformations on the training computation graph. We show this technique achieves substantial speedups on typical image and language models on Cloud TPUs, requiring no change to model code. This technique helps close the gap between traditionally expensive (ADAM) and cheap (SGD) optimizers, as they will only take a small part of training step time and have similar peak memory usage. It helped us to achieve state-of-the-art training performance in Google's MLPerf 0.6 submission.


# limitations
There are several limitations to be aware of when using FSDP:

- FSDP currently does not support gradient accumulation outside `no_sync()` when using CPU offloading. This is because FSDP uses the newly-reduced gradient instead of accumulating with any existing gradient, which can lead to incorrect results.
    
- FSDP does not support running the forward pass of a submodule that is contained in an FSDP instance. This is because the submodule’s parameters will be sharded, but the submodule itself is not an FSDP instance, so its forward pass will not all-gather the full parameters appropriately.
    
- FSDP does not work with double backwards due to the way it registers backward hooks.
    
- FSDP has some constraints when freezing parameters. For `use_orig_params=False`, each FSDP instance must manage parameters that are all frozen or all non-frozen. For `use_orig_params=True`, FSDP supports mixing frozen and non-frozen parameters, but it’s recommended to avoid doing so to prevent higher than expected gradient memory usage.
    
- As of PyTorch 1.12, FSDP offers limited support for shared parameters. If enhanced shared parameter support is needed for your use case, please post in [this issue](https://github.com/pytorch/pytorch/issues/77724).
    
- You should avoid modifying the parameters between forward and backward without using the `summon_full_params` context, as the modifications may not persist.

# step by step



1. **Model Initialization**

   * Model is wrapped in `FullyShardedDataParallel(model, ...)`.
   * Each GPU holds **only a shard** of the model’s parameters — not the full copy.

2. **Forward Pass**

   * When an input batch reaches a module:
     * FSDP **gathers** all parameter shards across GPUs → each GPU temporarily has the full weights for that layer.
     * Forward pass executes.
     * Then those full parameters are **freed (resharded)** to save memory.

3. **Backward Pass**

   * The same happens in reverse:

     * Gradients are computed locally.
     * Gradients are **sharded** again across GPUs.
     * Optional gradient accumulation happens efficiently.

4. **Optimizer Step**

   * Each GPU updates *its own shard* of parameters using its portion of gradients and optimizer states.
