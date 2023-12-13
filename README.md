# LLM-CRS

- **Paper**:  [Conversational Recommender System and Large Language Model Are Made for Each Other in E-commerce Pre-sales Dialogue]( https://aclanthology.org/2023.findings-emnlp.643.pdf)
- **Conference**: EMNLP 2023 findings
- **Authors**: Yuanxing Liu, Wei-Nan Zhang, Yifan Chen, Yuchi Zhang, Haopeng Bai, Fan Feng, Hengbin Cui, Yongbin Li, Wanxiang Che


## Abstract

E-commerce pre-sales dialogue aims to understand and elicit user needs and preferences for the items they are seeking so as to provide appropriate recommendations. Conversational recommender systems (CRSs) learn user representation and provide accurate recommendations based on dialogue context, but rely on external knowledge. Large language models (LLMs) generate responses that mimic pre-sales dialogues after fine-tuning, but lack domain-specific knowledge for accurate recommendations. Intuitively, the strengths of LLM and CRS in E-commerce pre-sales dialogues are complementary, yet no previous work has explored this. This paper investigates the effectiveness of combining LLM and CRS in E-commerce pre-sales dialogues, proposing two collaboration methods: CRS assisting LLM and LLM assisting CRS. We conduct extensive experiments on a real-world dataset of Ecommerce pre-sales dialogues. We analyze the impact of two collaborative approaches with two CRSs and two LLMs on four tasks of Ecommerce pre-sales dialogue. We find that collaborations between CRS and LLM can be very effective in some cases.


## Code Description

The released code is divided into three parts:

- data_processing: constructing training data for our models
- llm_tuning: the Training and evaluation code of LLMs
- UniMIND: the Training and evaluation code of CRSs

## Notice
- Note that we only include a few data samples in the code. To obtain the complete data set, please refer to the "DATASET ACCESS" section in the [U-NEED](https://dl.acm.org/doi/abs/10.1145/3539618.3591878) paper.
- If you have any questions, you can submit an issue or email yxliu@ir.hit.edu.cn

## Citation

If our code is helpful for your research, please cite us.
```
@inproceedings{liu-etal-2023-conversational,
    title = "Conversational Recommender System and Large Language Model Are Made for Each Other in {E}-commerce Pre-sales Dialogue",
    author = "Liu, Yuanxing  and
      Zhang, Weinan  and
      Chen, Yifan  and
      Zhang, Yuchi  and
      Bai, Haopeng  and
      Feng, Fan  and
      Cui, Hengbin  and
      Li, Yongbin  and
      Che, Wanxiang",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.643",
    pages = "9587--9605",
    abstract = "E-commerce pre-sales dialogue aims to understand and elicit user needs and preferences for the items they are seeking so as to provide appropriate recommendations. Conversational recommender systems (CRSs) learn user representation and provide accurate recommendations based on dialogue context, but rely on external knowledge. Large language models (LLMs) generate responses that mimic pre-sales dialogues after fine-tuning, but lack domain-specific knowledge for accurate recommendations. Intuitively, the strengths of LLM and CRS in E-commerce pre-sales dialogues are complementary, yet no previous work has explored this. This paper investigates the effectiveness of combining LLM and CRS in E-commerce pre-sales dialogues, proposing two collaboration methods: CRS assisting LLM and LLM assisting CRS. We conduct extensive experiments on a real-world dataset of E-commerce pre-sales dialogues. We analyze the impact of two collaborative approaches with two CRSs and two LLMs on four tasks of E-commerce pre-sales dialogue. We find that collaborations between CRS and LLM can be very effective in some cases.",
}
```
