import argparse
import json
import os
from numpy import mean
import numpy as np
import ast
from nltk import ngrams
from ast import literal_eval
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def acc_evaluate(predict, golden):
    '''
    :param predict: list of predict result. Each test example result is a string. eg ["系统提问", "用户需求", "功能需求:免清洗", "系统推荐"]
    :param golden: list of label. Each test example label is also a string. eg ["系统提问", "用户需求", "功能需求:免清洗", "系统推荐"]
    :return: mean of Precision, mean of Recall, mean of F1
    '''

    assert len(predict) == len(golden), "The lengths of predict results and labels are different."

    precisions = []
    recalls = []
    f1s = []
    
    for pred_str, gold_str in zip(predict, golden):
        pred_set = set(pred_str.split('；'))
        gold_set = set(gold_str.split('；'))

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        precision = tp / (tp + fp) if tp + fp != 0 else 0.0
        recall = tp / (tp + fn) if tp + fn != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {"Precision": np.mean(precisions), "Recall": np.mean(recalls), "F1": np.mean(f1s)}



def gen_evaluate(predict, golden):
    '''
    :param predict: list of predict result. Each test example result is a list of token.
    :param golden: list of label. Each test example label is also a list of token.
    :return: mean of Bleu@{1,2,3,4}, mean of Dist@{1,2,3,4}
    '''
    bleu_score = [[] for _ in range(4)]
    weights = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    dist_score = [[] for _ in range(4)]
    smoothie = SmoothingFunction().method4

    for p, g in zip(predict, golden):
        for k in range(4):
            bleu_score[k].append(sentence_bleu([g], p, weights=weights[k], smoothing_function=smoothie))

            k_grams = list(ngrams(p, k + 1))  # Convert to list before calculating length
            if len(k_grams) != 0:
                dist_score[k].append(len(set(k_grams)) / len(k_grams))
            else:
                dist_score[k].append(0.0)

    result = {}
    for k in range(4):
        result[f"Bleu@{k + 1}"] = mean(bleu_score[k])
        result[f"Dist@{k + 1}"] = mean(dist_score[k])

    return result


def recommend_evaluate(predict, golden, N_list=[1]):
    '''
    :param predict: List of strings, each string is a ranked list of item indices separated by commas
    :param golden:  List of strings, each string is a list of target item indices separated by commas
    :param N_list:  List of N values for evaluation
    :return: Dictionary of mean NDCG, MRR, and Hit for each N value
    '''

    eval_results = {}
    total_count = len(predict)

    for N in N_list:
        ndcg_list = []
        mrr_list = []
        hit_count = 0

        for top_item, golden_items in zip(predict, golden):
            top_item = top_item.split(",")
            golden_items = golden_items.split(",")  # Now this can contain multiple items

            max_ndcg_score = 0.0
            max_mrr_score = 0.0

            for golden_item in golden_items:
                if golden_item in top_item:
                    rank = top_item.index(golden_item)
                    if rank < N:
                        ndcg_score = 1.0 / np.log2(rank + 2)
                        mrr_score = 1.0 / (rank + 1)
                        hit_count += 1

                        max_ndcg_score = max(max_ndcg_score, ndcg_score)
                        max_mrr_score = max(max_mrr_score, mrr_score)

            ndcg_list.append(max_ndcg_score)
            mrr_list.append(max_mrr_score)

        eval_results[f"NDCG@{N}"] = np.mean(ndcg_list)
        eval_results[f"MRR@{N}"] = np.mean(mrr_list)
        eval_results[f"Hit@{N}"] = hit_count / total_count

    return eval_results




def read_result(args):
    res = []
    with open(f'./predict{args.stage}_{args.task}_new.txt') as f:
        for line in f:
            data = ast.literal_eval(line)
            res.append(data)
    pre = [i["predict"] for i in res]
    lab = [i["label"] for i in res]
    return pre, lab

def main(args):
    pre, lab = read_result(args)
    if args.task == 1 or args.task == 2:
        return acc_evaluate(pre,lab)
    elif args.task == 3:
        return recommend_evaluate(pre,lab)
    elif args.task == 4:
        return gen_evaluate(pre,lab) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--task', default=4, type=int)
    parser.add_argument('--stage', default="", type=str)
    args = parser.parse_args()
    res = main(args)

    directory = f'./path_to_eval_result{args.stage}_new/'
    os.makedirs(directory, exist_ok=True)
    with open(f'./path_to_eval_result{args.stage}_new/res{args.task}.txt',"w+",encoding="utf-8") as f:
        f.write(str(res))