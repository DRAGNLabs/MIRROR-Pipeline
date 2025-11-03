https://www.deepspeed.ai/
Microsoft. 
Created ZeRO. splits up weights, gradients, and optimizer states. [[FSDP]] based off step 3 of ZeRO. 

Another mechanism offloads the full model state to CPU or NVMe memory. very flexible

Good at compression and quantization. Pretty seamless parallelism, to the point of experimenting with automatic choice of how to parallelize depending on resources available. Some experiments 

used for:
- [Megatron-Turing NLG (530B)](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)
- [Jurassic-1 (178B)](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)
- [BLOOM (176B)](https://huggingface.co/blog/bloom-megatron-deepspeed)
- [GLM (130B)](https://github.com/THUDM/GLM-130B)
- [YaLM (100B)](https://github.com/yandex/YaLM-100B)
- [GPT-NeoX (20B)](https://github.com/EleutherAI/gpt-neox)
- [AlexaTM (20B)](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning)
- [Turing NLG (17B](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)
- [METRO-LM (5.4B)](https://arxiv.org/pdf/2204.06644.pdf)

Used by:
- Huggingface Transformers
- Huggingface Accelerate
- Pytorch Lightning
- MosaicML


