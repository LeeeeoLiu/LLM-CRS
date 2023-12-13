#!/bin/bash

python scripts/inference_hf.py \
    --base_model ./_checkpoint \
    --with_prompt \
    --predictions_file ./predict.txt \
    --data_file ./u_need


