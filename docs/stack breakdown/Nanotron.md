
[github](https://github.com/huggingface/nanotron)

Super simple, pure pytorch. Made by Huggingface, but not integrated into anything. 
Built for:
> Minimalistic large language model 3D-parallelism training

> We currently support the following features:
	- [x]  3D parallelism (DP+TP+PP)
	- [x]  Expert parallelism for MoEs
	- [x]  AFAB and 1F1B schedules for PP
	- [x]  Explicit APIs for TP and PP which enables easy debugging
	- [x]  ZeRO-1 optimizer
	- [x]  FP32 gradient accumulation
	- [x]  Parameter tying/sharding
	- [x]  Custom module checkpointing for large models
	- [x]  Spectral µTransfer parametrization for scaling up neural networks
	- [x]  Mamba example
	- [x]  CUDA event-based timing for accurate GPU performance measurement
   And we have on our roadmap:
	- [ ]  FP8 training
	- [ ]  ZeRO-3 optimizer (a.k.a FSDP)
	- [ ]  `torch.compile` support
	- [ ]  Ring attention
	- [ ]  Interleaved 1f1b schedule