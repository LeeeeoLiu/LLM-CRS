
import os
import json
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from nltk.util import ngrams
from collections import Counter

from transformers import Seq2SeqTrainer
from transformers.trainer import PredictionOutput, TRAINING_ARGS_NAME
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.tokenization_utils import PreTrainedTokenizer

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .config import FinetuningArguments

from .other import (
    get_logger,
    save_trainable_params,
    IGNORE_INDEX,
    FINETUNING_ARGS_NAME,
    PREDICTION_FILE_NAME
)


logger = get_logger(__name__)

@dataclass
class MyMetrics:
    def __init__(self, task, tokenizer):
        self.task = task
        self.tokenizer = tokenizer

    def mrr_score(y_true, y_pred):
        order = np.argsort(y_pred)[::-1]
        y_true = np.take(y_true, order)
        return np.sum([1. / (np.where(y_true == 1)[0][i] + 1) for i in range(len(y_true))]) / len(y_true)
    def hit_rate(y_true, y_pred, top_n=10):
        """
        y_true: list of true items
        y_pred: list of predicted items
        top_n: number of top items to consider in hit rate
        """
        top_n_pred = np.argsort(y_pred)[-top_n:]
        hits = [1 if true_item in top_n_pred else 0 for true_item in y_true]
        return np.mean(hits)

    def distinct_n(sentence, n):
        """
        Compute distinct-n for a single sentence.
        :param sentence: a list of words.
        :param n: int, ngram.
        :return: float, distinct-n score.
        """
        if len(sentence) == 0:
            # Prevent a zero division
            return 0.0
        distinct_ngrams = Counter(ngrams(sentence, n))
        return len(distinct_ngrams) / len(sentence)

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        preds, labels = eval_preds
        # print("preds:",len(preds))
        # print("labels:",len(labels))
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        score_dict = {}
        
        for pred, label in zip(preds, labels):
            hypothesis = list(jieba.cut(self.tokenizer.decode(pred, skip_special_tokens=True)))
            reference = list(jieba.cut(self.tokenizer.decode(label, skip_special_tokens=True)))
            print("hypothesis",hypothesis)
            print("reference",reference)
            if self.task == 1 or self.task == 2:
                precision = precision_score(labels, preds, average='micro')
                recall = recall_score(labels, preds, average='micro')
                f1 = f1_score(labels, preds, average='micro')
                score_dict = {"precision": precision, "recall": recall, "f1": f1}
            elif self.task == 3:
                ndcg = ndcg_score(reference, hypothesis)
                mrr = self.mrr_score(reference, hypothesis)
                # hit = hit_score(reference, hypothesis)
                score_dict = {"ndcg": ndcg, "mrr": mrr} #, "hit": 0.0}
            elif self.task == 4:
                bleu = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                dist = self.distinct_n(hypothesis,4)
                score_dict = {"bleu-4": bleu, "dist": dist}

        # take the average of the scores
        return {k: float(np.mean(v)) for k, v in score_dict.items()}