"""
Converting CRSs training data to fine-tune data for LLMs
"""

import os
import numpy as np
import json
import jsonlines
import pandas as pd
import argparse


TASKS = ['Understand', 'Elicit', 'Recommend', 'Response']
key_exclude = ['品牌', '一级类目', '叶子类目']

alphabet_list = [chr(i) for i in range(ord('A'), ord('A') + 20)]


def extract_user_need(dia_input):
    user_need_list = []
    for i in range(len(dia_input)-1):
        if dia_input[i].startswith("用户：") and dia_input[i+1].startswith("需求："):
            user_need_list.append(dia_input[i+1].lstrip("需求："))
    user_need_unit_list = []
    for item in user_need_list:
        units = item.split('；')
        for unit in units:
            if unit != '':
                unit_dict = {}
                unit_dict[unit.split('：')[0]] = unit.split('：')[1]
                user_need_unit_list.append(unit_dict)
    user_need_dict = {}
    for item in user_need_unit_list:
        for key in item:
            if  key not in user_need_dict:
                user_need_dict[key] = [item[key]]
            else:
                user_need_dict[key].append(item[key])

    return user_need_dict


def get_processed_input(dia_input):
    dia_list = []
    for sentence in dia_input:
        if sentence.startswith("需求："):
            pass
        else:
            dia_list.append(sentence)

    context = dia_list
    user_need_dict = extract_user_need(dia_input)

    return context, user_need_dict


def list2str_conversation(data):
    str_conversation = ''
    round_num = 0
    for i in range(0, len(data), 2):
        str_conversation += f'[Round {round_num}]\n'
        str_conversation += data[i] + '\n'
        if i+1 < len(data):  
            str_conversation += data[i+1] + '\n' # Add customer service response
        round_num += 1
    return str_conversation


def prepare_data_final(dataset_name, product_knowledge_base):
    with open(dataset_name, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    sample_id = 0
    for data in dataset:
        dia_domain = data["domain"]
        context, user_needs = get_processed_input(data["input"])

        if data["task_name"] == TASKS[0]:
            instruction = "根据{}售前对话，识别当前用户或者客服输入中涉及的商品相关的属性值和对应的属性。\n针对用户输入，需要识别出属性和属性值。客服输入中属性值可能为空。".format(
                dia_domain)
            input_llm = '售前对话：{}当前输入：{}'.format(list2str_conversation(context[:-1]), context[-1])
            output_llm = data["label"]

        if data["task_name"] == TASKS[1]:
            instruction = "根据{}售前对话，选择一系列的属性，来引导用户提供更多关于需求的偏好信息。结果中可以包含属性值，也可以不包含属性值。".format(dia_domain)
            input_llm = '售前对话：{}'.format(list2str_conversation(context))
            output_llm = data["label"]

        if data["task_name"] == TASKS[2]:
            instruction = "根据{}售前对话中用户表达的需求和偏好信息以及候选商品信息，从候选商品A-T中选择最有可能满足用户需求、偏好的商品推荐给用户。".format(dia_domain)
            select_items = data['candidate']
            item_infos = []

            for seq, item in enumerate(select_items):
                if item in product_knowledge_base:
                    item_infos.append("{}的{}".format(alphabet_list[seq], '，'.join(
                        "{}是{}".format(k, '、'.join(v)) for k, v in product_knowledge_base[item].items() if
                        k not in key_exclude)))
            input_llm= '售前对话：{}各候选商品对应的属性和属性值：{}'.format(list2str_conversation(context), '；'.join(item_infos))
            output_llm = chr(ord('A') + select_items.index(data['item_id']))

        if data["task_name"] == TASKS[3]:
            instruction = "根据{}售前对话中已获取的信息、引导用户需求的属性、满足用户需求的商品信息，生成回应用户需求且用户容易理解的通俗回复。".format(dia_domain)
            expressed_user_needs = ["{}：{}".format(k, '、'.join(v)) for k, v in user_needs.items()]
            elicit_attr = data['elicit_attr']
            if len(elicit_attr) != 0:
                user_needs_guide_attr = '\n引导用户需求的属性：{}'.format(elicit_attr)
            else:
                user_needs_guide_attr = ''

            recommended_items = data['rec_item_id'].split('；')

            if recommended_items[0] != '':
                item_infos = []
                for seq, item in enumerate(recommended_items):
                    if item in product_knowledge_base:
                        item_infos.append("商品{}满足用户需求，{}的{}".format(alphabet_list[seq], alphabet_list[seq],
                                                                             '，'.join(
                                                                                 "{}是{}".format(k, '、'.join(v)) for
                                                                                 k, v in
                                                                                 product_knowledge_base[item].items() if
                                                                                 k not in key_exclude)))
                recommended_items_str = '\n满足用户需求的商品信息：{}'.format('；'.join(item_infos))
            else:
                recommended_items_str = ''
            input_llm = '售前对话：{}已获取的用户需求偏好信息：{}{}{}'.format(list2str_conversation(context),
                                                                          '；'.join(expressed_user_needs),
                                                                          user_needs_guide_attr,
                                                                          recommended_items_str)
            output_llm = data["label"]

        data["instruction"] = instruction
        data["input_llm"] = input_llm
        data["output_llm"] = output_llm
        data["sample_id"] = sample_id
        sample_id += 1
    return dataset


parser = argparse.ArgumentParser(description='Converting CRSs training data to fine-tune data for LLMs')

parser.add_argument("--dataset_path", type=str, default='xx', help="Path to the json data constructed for training CRSs")
parser.add_argument("--save_path", type=str, default='xx', help="Path to the json data to be constructed for training LLMs")
parser.add_argument("--kg_file", type=str, default='kg_item_air_dataset_with_encrypt_0210.txt', help="File of the item mapping data, whose field contains item_id_encrypt, seller_id_encrypt, relation_name and value")


if __name__ == '__main__':
    
    args = parser.parse_args()

    df_kg = pd.read_csv(os.path.join(args.dataset_path, args.kg_file)).applymap(str)
    relation_name_list = df_kg['relation_name'].unique().tolist()
    relation_name_range = str(relation_name_list).replace('\'','').replace(' ','')

    product_knowledge_base = {}
    seller_products = {}

    with open(os.path.join(args.dataset_path, args.kg_file), 'r') as f:
        for line in f:
            item_id, seller_id, relation_name, value = line.strip().split(',')
            if item_id in product_knowledge_base:
                if relation_name in product_knowledge_base[item_id]:
                    product_knowledge_base[item_id][relation_name].append(value)
                else:
                    product_knowledge_base[item_id][relation_name] = [value]
            else:
                product_knowledge_base[item_id] = {relation_name: [value]}

            if seller_id in seller_products:
                seller_products[seller_id].append(item_id)
            else:
                seller_products[seller_id] = [item_id]
                
    data_file_name_list = ['train_sample.json', 'valid_sample.json', 'test_sample.json']
    data_type = ['train_sample', 'valid_sample', 'test_sample']
    for i, data_file_name in enumerate(data_file_name_list):

        result_list = prepare_data_final(os.path.join(args.dataset_path, data_file_name), product_knowledge_base)
        with open(os.path.join(args.save_path, data_type[i]+'.json'), 'w', encoding='utf-8') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=4)



