#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./src/train_sft.py \
    --do_train \
    --dataset  name_of_dataset\
    --finetuning_type lora \
    --output_dir path_to_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16 \
