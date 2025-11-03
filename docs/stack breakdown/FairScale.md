Pytorch extension library from Facebook research that originally implemented FSDP into pytorch. 



From the [blog post](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) introducing [[FSDP]] into pytorch:
> Recent approaches like DeepSpeed ZeRO and FairScale’s Fully Sharded Data Parallel allow us to break this barrier by sharding a model’s parameters, gradients and optimizer states across data parallel workers while still maintaining the simplicity of data parallelism.
> With PyTorch 1.11 we’re adding native support for Fully Sharded Data Parallel (FSDP)...
> 
> In future PyTorch versions, we’re going to enable users to seamlessly switch between DDP, ZeRO-1, ZeRO-2 and FSDP flavors of data parallelism

and you can see from its github page that it's influenced pretty heavily by several other projects, like [[Megatron-LM]]. 
>     FairScale is licensed under the [BSD-3-Clause License](https://github.com/facebookresearch/fairscale/blob/main/LICENSE).
	fairscale.nn.pipe is forked from [torchgpipe](https://github.com/kakaobrain/torchgpipe), Copyright 2019, Kakao Brain, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).
	fairscale.nn.model_parallel is forked from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), Copyright 2020, NVIDIA CORPORATION, licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).
	fairscale.optim.adascale is forked from [AdaptDL](https://github.com/petuum/adaptdl), Copyright 2020, Petuum, Inc., licensed under [Apache License](http://www.apache.org/licenses/LICENSE-2.0).
	fairscale.nn.misc.flatten_params_wrapper is forked from [PyTorch-Reparam-Module](https://github.com/SsnL/PyTorch-Reparam-Module), Copyright 2018, Tongzhou Wang, licensed under [MIT License](https://github.com/SsnL/PyTorch-Reparam-Module/blob/master/LICENSE).


# current usage
Meta hasn’t published full details yet, but interviews state they moved from FairScale to native PyTorch [[FSDP]] with tensor parallelism.