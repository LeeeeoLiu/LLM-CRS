import os
import json as json
import pandas as pd
import random
import argparse

random.seed(1234)
candidates_num = 20


def get_candicates(sid, seller_id, rec_item_id_list, seq_no, df_kg):
    """
    According to the recommended item in the dialogue, build a recommended list for it, with 1 recommended item  and 19 other items
    If the items in the corresponding store are insufficient, sample from other stores
    """

    res = []
    items_list = df_kg['item_id_encrypt'].unique().tolist()
    # Get items of the same store 
    seller_item_list = df_kg[df_kg['seller_id_encrypt'] == seller_id]['item_id_encrypt'].unique().tolist()
    for item_id in rec_item_id_list:
        if item_id not in items_list:
            print(f'None!:{item_id}')

        try:
            seller_item_list.remove(item_id)
        except:
            print(f'Error!:{item_id}')

    # Build the candidate list of recommedation
    for rec_item_id in rec_item_id_list:
        
        # Randomly select 19 non-repeating items
        item_candidates_list = []
        if len(seller_item_list) < candidates_num-1:
            # Handling situations where there are less than 20 items in the store
            
            item_candidates_list.append(rec_item_id)
            item_candidates_list = item_candidates_list + seller_item_list

            # Extract type information of recommended item
            item_type = df_kg.loc[
                (df_kg['item_id_encrypt'] == rec_item_id) & (df_kg['relation_name'] == '一级类目'), 'value'].values[
                0]
            
            # According to the found value and relation_name, extract item id from items of the same type
            type_item_list = df_kg.loc[(df_kg['value'] == item_type) & (
                    df_kg['relation_name'] == '一级类目'), 'item_id_encrypt'].unique().tolist()
            for item_id in item_candidates_list:
                if item_id in type_item_list:
                    type_item_list.remove(item_id)

            if len(type_item_list) >= candidates_num - len(item_candidates_list):
                item_candidates_list = random.sample(type_item_list, candidates_num - len(item_candidates_list)) + item_candidates_list
            else:
                # Handle the situation that there are less than 20 items in the corresponding store
                print(f'Wrong!:{item_id}')
                item_candidates_list = type_item_list + item_candidates_list
                temp_items_list = items_list.copy()
                for item_id in item_candidates_list:
                    if item_id in temp_items_list:
                        temp_items_list.remove(item_id)
                item_candidates_list = random.sample(temp_items_list,candidates_num - len(item_candidates_list)) + item_candidates_list
        else:
            item_candidates_list = random.sample(seller_item_list, candidates_num-1)
            item_candidates_list.append(rec_item_id)
            
        # Disrupt the order of item_candidates_list
        random.shuffle(item_candidates_list)
        # Add item_candidates_list to the return value
        res.append(item_candidates_list)

    return res


def dia_data_process(dialog_path, new_path, df_kg):
    with open(dialog_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i, element in enumerate(data):
        seller_id = element['sellerid']
        sid = element['sid']

        for dia_turn in element['dialogue']:
            rec_item_id_list = dia_turn['rec_item_id']
            seq_no = dia_turn['seq_no']
            candicates_result = get_candicates(sid, seller_id, rec_item_id_list, seq_no, df_kg)

            dia_turn['rec_item_candidates'] = candicates_result

            for i,item_id in enumerate(rec_item_id_list):
                if len(candicates_result[i]) != candidates_num:
                    print('number error!!!')
                if item_id not in candicates_result[i]:
                    print('select error!!!')

    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


parser = argparse.ArgumentParser(description='Adding candidate items to the recommendation stage in the raw dialog data')

parser.add_argument("--path_ori", type=str, default='xx', help="Path where raw dialog data is stored")
parser.add_argument("--path_new", type=str, default='xx', help="Path where new dialog data to be stored")
parser.add_argument("--kg_file", type=str, default='kg_item_air_dataset_with_encrypt_0210.txt', help="File of the item mapping data, whose field contains item_id_encrypt, seller_id_encrypt, relation_name and value")


if __name__ == '__main__':
    args = parser.parse_args()
    kg_path = os.path.join(args.path_ori, args.kg_file)
    df_kg = pd.read_csv(kg_path).applymap(str)

    dialog_train = 'air_train_dataset_0210_encrypt.json'
    dialog_dev = 'air_dev_dataset_0210_encrypt.json'
    dialog_test = 'air_test_dataset_0210_encrypt.json'

    dia_data = [dialog_train, dialog_dev, dialog_test]
    for data_name in dia_data:
        print('#' * 50)
        print(data_name)
        dia_data_process(os.path.join(args.path_ori, data_name), os.path.join(args.path_new, data_name), df_kg)
