from numpy import mean


def acc_evaluate(predict, golden):
    """
    :param predict: list of predict result. Each test example result is a set.
    :param golden: list of label. Each test example label is also a set.
    :return: mean of Precision, mean of Recall, mean of F1

    eg. For task1.
    predict: [{("年龄", "20多岁"), ("肤质", "干")}, {("价格", "便宜一点的"), ("品类", "跑鞋")}]
    golden: [{("年龄", "20多岁"), ("肤质", "有点干")}, {("性别", "老公"), ("价格", "便宜一点的"), ("品类", "跑鞋")}]
    output: {'Precision': 0.75, 'Recall': 0.5833333333333333, 'F1': 0.65}

    For task2
    predict: [{"肤质"}, {"肌肤问题"}, set()]
    golden: [{"年龄", "肤质"},{"肤质", "功效"}, {"功效"}]
    output: {'Precision': 0.3333333333333333, 'Recall': 0.16666666666666666, 'F1': 0.2222222222222222}
    """
    if len(predict) == 0:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    precision_list = []
    recall_list = []
    f1_list = []
    for p, g in zip(predict, golden):
        tp = len(p & g)
        precision_list.append(tp / len(p) if len(p) != 0 else 0.0)
        recall_list.append(tp / len(g) if len(g) != 0 else 0.0)
        f1_list.append(
            2 * precision_list[-1] * recall_list[-1] / (precision_list[-1] + recall_list[-1]) if tp != 0 else 0.0)

    return {"Precision": mean(precision_list), "Recall": mean(recall_list), "F1": mean(f1_list)}


def domain_acc_evaluate(predict, golden, domain_data):
    result = {"所有行业": acc_evaluate(predict, golden)}
    for domain in ["美妆行业", "手机行业", "大家电行业", "服装行业", "鞋类行业"]:
        filter_predict = []
        filter_golden = []
        for index, d in enumerate(domain_data):
            if d == domain:
                filter_predict.append(predict[index])
                filter_golden.append(golden[index])

        result[domain] = acc_evaluate(filter_predict, filter_golden)
    return result
