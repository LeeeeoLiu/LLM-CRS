from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from numpy import mean

def gen_evaluate(predict, golden):
    '''
    :param predict: list of predict result. Each test example result is a list of token.
    :param golden: list of label. Each test example label is also a list of token.
    :return: mean of Bleu@{1,2,3,4}, mean of Dist@{1,2,3,4}
    '''
    result = {}
    if len(predict) == 0:
        for k in range(4):
            result[f"Bleu@{k + 1}"] = 0.0
            result[f"Dist@{k + 1}"] = 0.0
        return result
        
    bleu_score = [[],[],[],[]]
    weights = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    dist_score = [[],[],[],[]]
    for p, g in zip(predict, golden):
        p_list = list(p)
        g_list = list(g)

        for k in range(4):
            bleu_score[k].append(sentence_bleu([g_list], p_list, weights=weights[k]))

            k_grams = list(ngrams(p_list, k+1))
            if len(k_grams) != 0:
                dist_score[k].append(len(set(k_grams)) / len(k_grams))
            else:
                dist_score[k].append(0.0)

    for k in range(4):
        result[f"Bleu@{k + 1}"] = mean(bleu_score[k])
        result[f"Dist@{k + 1}"] = mean(dist_score[k])

    return result

def domain_gen_evaluate(predict, golden, domain_data):
    result = {"所有行业": gen_evaluate(predict, golden)}
    for domain in ["美妆行业","手机行业","大家电行业","服装行业","鞋类行业"]:
        filter_predict = []
        filter_golden = []
        for index, d in enumerate(domain_data):
             if d == domain:
                 filter_predict.append(predict[index])
                 filter_golden.append(golden[index])
    
        result[domain] = gen_evaluate(filter_predict,filter_golden)
    return result