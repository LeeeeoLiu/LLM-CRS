#!/bin/bash

python scripts/merge_llama_with_chinese_lora.py \
    --base_model ./llama-7b-hf \
    --lora_model ./checkpoint_sta2_un/sft_lora_model \
    --output_type huggingface \
    --output_dir out_checkpoint 