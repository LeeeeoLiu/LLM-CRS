#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python ./src/train.py \
    --do_predict \
    --dataset name_of_dataset \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100000 \
    --predict_with_generate \
    --fp16 