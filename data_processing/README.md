# Data Processing

Constructing training data for LLMs and CRSs for two stages. In this part, stage 1 means the modeling way without collaboration and stage 2 means collaboration modeling way with collaboration. 

### rec_data_process.py

This program adds a candidate set of items for the inference and evaluation process of the recommendation task on the basis of the original data, and the processed data can be used for stage 1 training of CRSs

Useage:
```bash
python rec_data_process.py  \
    --path_ori ori_data_dir \
    --path_new data_save_dir
```

### unified2ft.py

Based on the data processed in the previous step, we can build standard data to train CRSs (in the UniMIND code part). Then we can construct the fine-tune data for stage 1 of LLMs for Tasks 1-4. 

Useage:
```bash
python unified2ft.py  \
    --dataset_path crs_training_data_dir \
    --save_path data_save_dir
```


### crs_s2_data_construct.py
Merge the results inferenced by LLMs in stage 1  with the original data to construct the training data for stage 2 of CRSs

Useage:
```bash
python crs_s2_data_construct.py  \
    --llm_train_data_path stage1_train_data_dir_for_LLMs \
    --llm_res_data_path stage1_result_data_dir_of_LLMs \
    --llm_type chatglm_or_alpaca \
    --data_save_path stage2_train_data_dir_for_CRSs
```


### llm_s2_data_construct.py

Merge the results inferenced by CRSs in stage 1 with the original data to construct the training data for stage 2 of LLMs

Useage:
```bash
python llm_s2_data_construct.py  \
    --crs_res_data_path stage1_result_data_dir_of_CRSs \
    --data_save_path stage2_train_data_dir_for_LLMs \
    --crs_type cpt_or_bart 
```
