import numpy as np


def recommend_evaluate(predict, golden, k_list):
    '''
    :param predict: List of ranked topN item index. Note category filter or store filter should be done before pass in.
     This method only need top50 item id.
    :param golden: List of target item id.
    :param k_list: List of k in [NDCG@k, MRR@k, Hit@k]. Note: len(predict[0]) >= max(k_list)
    :return: mean of NDCG@{k}, mean of MRR@{k}, mean of Hit@{k}
    '''
    results = {}
    
    if len(predict) == 0:
        for k in k_list:
            results[f"NDCG@{k}"] = 0.0
            results[f"MRR@{k}"] = 0.0
            results[f"Hit@{k}"] = 0.0
        return results
    
    total_count = len(predict)
    for k in k_list:
        ndcg_k_list = []
        mrr_k_list = []
        hit_k_count = 0

        for top_item, golden_item in zip(predict, golden):
            ndcg_k_score = 0.0
            mrr_k_score = 0.0
            if golden_item in top_item:
                rank = top_item.index(golden_item)

                if rank < k:
                    hit_k_count += 1
                    ndcg_k_score = 1.0 / np.log2(rank + 2)
                    mrr_k_score = 1.0 / (rank + 1)
            ndcg_k_list.append(ndcg_k_score)
            mrr_k_list.append(mrr_k_score)

        results[f"NDCG@{k}"] = np.mean(ndcg_k_list)
        results[f"MRR@{k}"] = np.mean(mrr_k_list)
        results[f"Hit@{k}"] = hit_k_count / total_count

    return results

def domain_recommend_evaluate(predict, golden, k_list,domain_data):
    result = {"所有行业": recommend_evaluate(predict, golden, k_list)}
    for domain in ["美妆行业","手机行业","大家电行业","服装行业","鞋类行业"]:
        filter_predict = []
        filter_golden = []
        for index, d in enumerate(domain_data):
             if d == domain:
                 filter_predict.append(predict[index])
                 filter_golden.append(golden[index])
    
        result[domain] = recommend_evaluate(filter_predict,filter_golden, k_list)
    return result