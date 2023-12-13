## Model download

### Notice to Users (Required Reading)）

Facebook's official release of [LLaMA model prohibits commercial use](https://github.com/facebookresearch/llama), and there is no official open source model weight (although there are already many third-party download addresses online). In order to comply with the corresponding license, it is currently not possible to publish the full model weights, please understand (the same is true for foreign countries at present). **What is released here is the LoRA weight**, which can be understood as a "patch" on the original LLaMA model, and the two can be combined to obtain the full copyright weight. The following Chinese LLaMA/Alpaca LoRA model cannot be used alone and needs to be paired with [Original LLaMA Model](https://github.com/facebookresearch/llama). Please refer to the [Merge Model]( #MergeModel) steps given in this project to reconstruct the model.

### Chinese Alpaca Model

Chinese Alpaca model further uses instruction data to fine-tune on the basis of the above Chinese LLaMA model. **If you want to experience ChatGPT-like conversational interaction, please use the Alpaca model instead of the LLaMA model.**

| Model name                  | training data |                   Model refactoring<sup>[1]</sup>                   | size<sup>[2]</sup> |                    LoRA download<sup>[3]</sup>                    |
| :------------------------ | :------: | :--------------------------------------------------------: | :----------------: | :----------------------------------------------------------: |
| Chinese-Alpaca-7B         |   2M instructions  |                        original LLaMA-7B                        |        790M        | [[Baidu disk]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>[[Google Drive]](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing) |

### Model Hub

All of these models can be downloaded on the Model Hub and the Chinese LLaMA or Alpaca LoRA models can be invoked using [transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft). The following model invocation names refer to model names specified in `.from_pretrained()`.

| Model name                  | Model repository name                            |                             Link                             |
| ----------------------- | :-------------------------------------- | :----------------------------------------------------------: |
| Chinese-Alpaca-7B       | ziqingyang/chinese-alpaca-lora-7b       | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b) 

### Footnotes and other notes

**[1]** Refactoring requires the original LLaMA model, [go to the LLaMA project to apply for use](https://github.com/facebookresearch/llama) or refer to this [PR](https://github.com/facebookresearch/llama/pull/73/files). Due to copyright issues, download links cannot be provided for this project.

**[2]** The reconstructed model is larger than the original LLaMA of the same magnitude (mainly because of the expanded vocabulary).


The file directories in the compressed package are as follows (take Chinese-LLaMA-7B as an example):

```
chinese_llama_lora_7b/
  - adapter_config.json		
  - adapter_model.bin		
  - special_tokens_map.json	
  - tokenizer_config.json	
  - tokenizer.model		   
```


## MergeModel

 [GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/模型合并与转换)


## Other more detailed operational information is all from the following repository
[[repository](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/)]
