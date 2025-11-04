
According to [HF](https://huggingface.co/docs/transformers/v4.57.0/perf_train_gpu_many#pipeline-parallelism)

>Pipeline parallelism is conceptually very similar to model parallelism, but it’s more efficient because it reduces the amount of idle GPU time. Instead of waiting for each GPU to finish processing a batch of data, pipeline parallelism creates _micro-batches_ of data. As soon as one micro-batch is finished, it is passed to the next GPU. This way, each GPU can concurrently process part of the data without waiting for the other GPU to completely finish processing a mini batch of data.

>Pipeline parallelism shares the same advantages as model parallelism, but it optimizes GPU utilization and reduces idle time. But pipeline parallelism can be more complex because models may need to be rewritten as a sequence of [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) modules and it also isn’t possible to completely reduce idle time because the last `forward` pass must also wait for the `backward` pass to finish.


Still in alpha stage in [pytorch](https://docs.pytorch.org/docs/stable/distributed.pipelining.html). Exists in Megatron already. DeepSpeed has its own implementation, but it's not compatible with zero-2 or zero-3, using gradient accumulation?

## What is `torch.distributed.pipelining`?

While promising for scaling, pipelining is often difficult to implement because it needs to **partition the execution** of a model in addition to model weights. The partitioning of execution often requires intrusive code changes to your model. Another aspect of complexity comes from **scheduling micro-batches in a distributed environment**, with **data flow dependency** considered.

The `pipelining` package provides a toolkit that does said things **automatically** which allows easy implementation of pipeline parallelism on **general** models.

It consists of two parts: a **splitting frontend** and a **distributed runtime**. The splitting frontend takes your model code as-is, splits it up into “model partitions”, and captures the data-flow relationship. The distributed runtime executes the pipeline stages on different devices in parallel, handling things like micro-batch splitting, scheduling, communication, and gradient propagation, etc.

Overall, the `pipelining` package provides the following features:

- Splitting of model code based on simple specification.
    
- Rich support for pipeline schedules, including GPipe, 1F1B, Interleaved 1F1B and Looped BFS, and providing the infrastructure for writing customized schedules.
    
- First-class support for cross-host pipeline parallelism, as this is where PP is typically used (over slower interconnects).
    
- Composability with other PyTorch parallel techniques such as data parallel (DDP, FSDP) or tensor parallel. The [TorchTitan](https://github.com/pytorch/torchtitan) project demonstrates a “3D parallel” application on the Llama model.