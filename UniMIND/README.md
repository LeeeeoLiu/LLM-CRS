# Environment
Python >=3.8
```
pip install -r requirements.txt
```

# Run
```bash
python pretrain_crs.py
```

Some important args:
1. --sample_random: Use only 1/20 data during train, valid, test.
2. --no_wandb: no wandb
3. --test: no training and set load_model_path in pretrain.py
4. --recommend_finetune: default true, will train task3 firstly, then all tasks.
4. --llm [chatglm|alpaca]: Add llm output in train/valid/test data
