


### Data preparation

For the format of the dataset file, refer to the contents of the `data/u_need` folder. When building a custom dataset, update the `data/dataset_info.json` file, which is formatted as follows in the following code.
```json
"u_need": {
        "file_name": "u_need/train.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": ""
        }
    }
```



### Single-GPU fine-tuning training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
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
    --fp16 
```
 

### Metric evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python /src/eval.py \
  --task \
  --stage 
```




### Model prediction
```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --do_predict \
    --dataset name_of_dataset \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100000 \
    --predict_with_generate \
    --fp16 
```


### 硬件需求

|      methods     | batch size | mode | GPUmem | speed |
| ---------------- | ---------- | ---- | ------ | ----- |
| LoRA (r=8)       |     16     | FP16 |  28GB  | 8ex/s |
| LoRA (r=8)       |     8      | FP16 |  24GB  | 8ex/s |
| LoRA (r=8)       |     4      | FP16 |  20GB  | 8ex/s |
| LoRA (r=8)       |     4      | INT8 |  10GB  | 8ex/s |
| P-Tuning (p=16)  |     4      | FP16 |  20GB  | 8ex/s |
| P-Tuning (p=16)  |     4      | INT8 |  16GB  | 8ex/s |
| P-Tuning (p=16)  |     4      | INT4 |  12GB  | 8ex/s |
| Freeze (l=3)     |     4      | FP16 |  24GB  | 8ex/s |
| Freeze (l=3)     |     4      | INT8 |  12GB  | 8ex/s |


> Note: `r` is the LoRA dimension size, `p` is the prefix vocabulary size, `l` is the number of fine-tuning layers, and `ex/s` is the number of samples trained per second. The `gradient_accumulation_steps` parameter is set to `1`. The above results are from a single Tesla V100 GPU and are for informational purposes only.

