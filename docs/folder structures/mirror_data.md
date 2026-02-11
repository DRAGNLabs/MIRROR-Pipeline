# Folder Structure of ~/nobackup/autodelete/mirror_data

If you're looking for checkpoints, configs, or other stored files, here is where to find it.

mirror_data
-- datasets
    -- Salesforce
        -- wikitext
-- models
    -- meta-llama
        -- Llama-3.2-1B
        -- Llama-3.2-1B-Instruct
    -- openai-community
        -- gpt2
-- tokenizers
    -- meta-llama
        -- Llama-3.2-1B
        -- Llama-3.2-1B-Instruct
    -- openai-community
        -- gpt2
-- training_runs

# datasets
Contains dataset source folders (e.g. Salesforce), which contain dataset folders (e.g. wikitext).

# models
Contains model family folders (e.g. meta-llama), which contain specific model folders (e.g. Llama-3.2-1B-Instruct). Each model folder contains the model's config.json, generation_config.json, and model.safetensors files.

# tokenizers
Contains model family folders (e.g. meta-llama), which contain specific model folders (e.g Llama-3.2-1B-Instruct). Each model folder contains the model's tokenizer.json and tokenizer_config.json files.

# training_runs
Contains a folder with logs for each submitted training run.