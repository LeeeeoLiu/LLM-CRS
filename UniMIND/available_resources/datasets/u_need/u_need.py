import copy
import json
import os
import pickle
from random import randint

import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast


class DataProcess:
    act_tag2special_token = {"用户需求": ["[user]", "[understand]"],
                             "系统提问": ["[elicit]", "[system]", "[understand]"],
                             "用户回答": ["[user]", "[understand]"],
                             "系统推荐": ["[recommend]", "[system]", "[understand]"],
                             "系统解释": ["[system]", "[understand]"],
                             "用户反馈": ["[user]", "[understand]"]}
    task_config = {"Understand": {"permit": ("[user]", "[understand]", "[system]"),
                                  "flag": "[understand]",
                                  "prompt": "。请识别前一句中的关键词并给出分类，以'属性：属性值'的方式，使用'；'连接："},
                   "Elicit": {"permit": ("[user]", "[system]", "[understand]", "[elicit]"),
                              "flag": "[elicit]",
                              "prompt": "。请基于前文给出合适的提问属性集合，以获取更多的需求信息："},
                   "Recommend": {"permit": ("[user]", "[system]", "[recommend]", "[understand]"),
                                 "flag": "[recommend]",
                                 "prompt": "",
                                 },
                   "Response": {"permit": ("[user]", "[system]", "[understand]"),
                                "flag": "[system]",
                                "prompt": "。生成系统回复："}}

    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        # static map for dataset
        self.candidate = {}
        self.dialogues = []
        self.samples = {"Understand": [], "Elicit": [], "Recommend": [], "Response": []}
        self.text_samples = []
        self.llm_output = {}

    def check_and_process(self):
        save_dir = os.path.join(self.args.data_root_path, "saved_data")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(f"{save_dir}/item_id2index.pkl.pkl"):
            self.dump_map_data()
        if not os.path.exists(os.path.join(save_dir, f"{self.mode}_samples.pkl")):
            self.load_llm_data()
            self.get_data()
            self.get_sample()
            return self.samples
        else:
            self.samples = pickle.load(open(os.path.join(save_dir, f"{self.mode}_samples.pkl"), "rb"))
            print("Load data successful.")
            return self.samples

    def dump_map_data(self):
        item_id_index_dict = {}
        kg_file = open(os.path.join(self.args.data_root_path,
                                    "item_kg.txt"), "r", encoding="UTF-8")
        lines = kg_file.readlines()
        kg_file.close()
        for line in lines[1:]:
            item_id, seller_id, attribute, value = line[:-1].split(",")
            if item_id not in item_id_index_dict.keys():
                item_id_index_dict[item_id] = len(item_id_index_dict)

        save_dir = os.path.join(self.args.data_root_path, "saved_data")
        pickle.dump(item_id_index_dict, open(os.path.join(save_dir, "item_id2index.pkl"), "wb"))

    def load_llm_data(self):
        load_path = os.path.join(self.args.data_root_path, "prompt4LLM", f"{self.mode}_sample.json")
        samples = json.load(open(load_path, "r", encoding="UTF-8"))

        for sample in samples:
            sid = sample["sid"]
            if sid not in self.llm_output.keys():
                self.llm_output[sid] = []

            if sample["task_name"] == "Recommend":
                chatglm_output = sample["candidate"][ord(sample["chatglm_predict_result"]) - ord("A")]

                try:
                    alpaca_output = sample["candidate"][ord(sample["alpaca_predict_result"]) - ord("A")]
                except:
                    alpaca_output = sample["candidate"][randint(0, 19)]

            else:
                chatglm_output = sample["chatglm_predict_result"]
                alpaca_output = sample["alpaca_predict_result"]

            self.llm_output[sid].append({
                "sample_id": sample["sample_id"],
                "sample_index_in_dialogue": sample["sample_index_in_dialogue"],
                "chatglm_output": chatglm_output,
                "alpaca_output": alpaca_output,
            })

    def get_data(self):
        dialogue_path = os.path.join(self.args.data_root_path, f"{self.mode}_dialogue.json")
        raw_dialogue_data = json.load(open(dialogue_path, "r", encoding="UTF-8"))

        for session in raw_dialogue_data:
            dialogue = {"user_id": session["userid"], "sid": session["sid"], "seller_id": session["sellerid"],
                        "domain": session["domain"], "llm_output": self.llm_output[session["sid"]]}
            candidates = {}
            for utterance in session["dialogue"]:
                if len(utterance["rec_item_id"]) > 0:
                    for index, item_id in enumerate(utterance["rec_item_id"]):
                        candidates[item_id] = utterance["rec_item_candidates"][index]
            dialogue["candidates"] = candidates

            sequence = []
            former_act_tag = None
            for utterance in session["dialogue"]:
                special_token_list = DataProcess.act_tag2special_token[utterance["act_tag"]]

                repeat_flag = former_act_tag == utterance["act_tag"]
                if utterance["act_tag"] == "系统解释" and former_act_tag == "系统推荐":
                    repeat_flag = True

                for special_token in special_token_list:
                    if special_token == "[user]":
                        value = [utterance["send_content"]]
                    elif special_token == "[understand]":
                        value = [a["key"] + "：" + a["value"] for a in utterance["attributes"]]
                    elif special_token == "[elicit]":
                        value = set([a["key"] for a in utterance["attributes"]])
                    elif special_token == "[system]":
                        value = [utterance["send_content"].replace("仅发送商品链接", "")]
                    elif special_token == "[recommend]":
                        value = utterance["rec_item_id"]
                    else:
                        assert False

                    if repeat_flag:
                        index = special_token_list.index(special_token) - len(special_token_list)
                        if special_token == "[elicit]":
                            sequence[index]["value"].update(value)
                        else:
                            sequence[index]["value"].extend(value)
                    else:
                        sequence.append({"special_token": special_token, "value": value})
                former_act_tag = utterance["act_tag"]

            for index, node in enumerate(sequence):
                special_token, value = node["special_token"], node["value"]
                if special_token in ["[user]", "[system]"]:
                    sequence[index]["value"] = " ".join(value)
                else:
                    sequence[index]["value"] = "；".join(value)

            dialogue["sequence"] = sequence
            self.dialogues.append(dialogue)

    @staticmethod
    def dialogue2sample(dialogue):
        all_samples = {}
        sample_index = 0

        for task in ["Understand", "Elicit", "Recommend", "Response"]:
            task_config = DataProcess.task_config[task]
            permit_special_token = task_config["permit"]
            flag_special_token = task_config["flag"]
            prompt = task_config["prompt"]

            input_output_sequences = []
            sequence = []
            former_special_token = None
            former_value = None
            for index, node in enumerate(dialogue["sequence"]):
                special_token, value = node["special_token"], node["value"]
                if special_token in permit_special_token:
                    if flag_special_token == special_token:
                        inputs = sequence.copy()
                        if task == "Response" and former_special_token in ["[recommend]", "[elicit]"]:
                            inputs.append(former_special_token + former_value)
                        inputs.append(prompt)
                        if task != "Recommend":
                            input_output_sequences.append((inputs, value))
                        else:
                            for item_id in value.split("；"):
                                input_output_sequences.append((inputs, item_id))
                    sequence.append(special_token + value)
                former_special_token = special_token
                former_value = value

            samples = []
            llm_output = dialogue["llm_output"]
            for in_seq, out_seq in input_output_sequences:
                if out_seq.replace(" ", "") == "":
                    continue

                sample = {
                    "input": in_seq,
                    "domain": dialogue["domain"],
                    "user_id": dialogue["user_id"],
                    "sid": dialogue["sid"],
                    "seller_id": dialogue["seller_id"],
                    "task_name": task,
                    "sample_index_in_dialogue": sample_index,
                    "sample_id": -1,
                }
                sample_index += 1

                if task in ["Understand", "Elicit", "Response"]:
                    sample["label"] = out_seq
                    sample["alpaca_input"] = sample["input"]
                    sample["chatglm_input"] = sample["input"]
                    for output in llm_output:
                        if output["sample_index_in_dialogue"] == sample["sample_index_in_dialogue"]:
                            sample["alpaca_input"] = sample["input"][:-1] + \
                                                     ["[LLM]" + output["alpaca_output"]] + \
                                                     [sample["input"][-1]]
                            sample["chatglm_input"] = sample["input"][:-1] + \
                                                      ["[LLM]" + output["chatglm_output"]] + \
                                                      [sample["input"][-1]]
                            sample["sample_id"] = output["sample_id"]
                            break
                elif task == "Recommend":
                    sample["item_id"] = out_seq
                    sample["candidate"] = dialogue["candidates"][out_seq]
                    sample["alpaca_item_id"] = out_seq
                    sample["chatglm_item_id"] = out_seq
                    for output in llm_output:
                        if output["sample_index_in_dialogue"] == sample["sample_index_in_dialogue"]:
                            sample["alpaca_item_id"] = output["alpaca_output"]
                            sample["chatglm_item_id"] = output["chatglm_output"]
                            sample["sample_id"] = output["sample_id"]
                            break
                else:
                    print("Evaluate or other task is not available.")
                    assert False
                samples.append(sample)
            all_samples[task] = samples
        return all_samples

    def get_sample(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.args.base_model_path,
                                                      additional_special_tokens=self.args.special_tokens)
        # separator = "[SEP]"
        separator = ""
        for dialogue in tqdm(self.dialogues):
            all_samples = DataProcess.dialogue2sample(dialogue)

            for task in ["Understand", "Elicit", "Recommend", "Response"]:
                samples = all_samples[task]
                for sample in samples:
                    # For dataset visualization
                    merge_sample = copy.deepcopy(sample)
                    if task in ["Understand", "Elicit", "Response"]:
                        merge_sample.pop("alpaca_input")
                        merge_sample.pop("chatglm_input")
                    elif task == "Recommend":
                        merge_sample.pop("alpaca_item_id")
                        merge_sample.pop("chatglm_item_id")
                    else:
                        print("Evaluate or other task is not available.")
                        assert False
                    merge_sample["input"] = []
                    temp_input = sample["input"]
                    for i in temp_input:
                        if i[0:6] == "[user]" or i[0:8] == "[system]" or i[0:12] == "[understand]":
                            merge_sample["input"].append(
                                i.replace("[user]", "用户：").replace("[system]", "客服：").replace("[understand]",
                                                                                                  "需求："))
                    if task == "Response":
                        merge_sample["elicit_attr"] = ""
                        merge_sample["rec_item_id"] = ""

                        if temp_input[-2][0:8] == "[elicit]":
                            merge_sample["elicit_attr"] = temp_input[-2].replace("[elicit]", "")
                        elif temp_input[-2][0:11] == "[recommend]":
                            merge_sample["rec_item_id"] = temp_input[-2].replace("[recommend]", "")

                    self.text_samples.append(merge_sample)

                    # For afterward dataloader and training
                    if task in ["Understand", "Elicit", "Response"]:
                        tokenized_input = tokenizer(separator.join(sample["input"]), padding="max_length",
                                                    truncation=True, max_length=512, return_tensors="pt",
                                                    return_attention_mask=True)
                        tokenized_alpaca_input = tokenizer(separator.join(sample["alpaca_input"]),
                                                           padding="max_length", max_length=512,
                                                           truncation=True, return_tensors="pt")
                        tokenized_chatglm_input = tokenizer(separator.join(sample["chatglm_input"]),
                                                            padding="max_length", max_length=512,
                                                            truncation=True, return_tensors="pt")
                        tokenized_label = tokenizer(sample["label"], padding="max_length", max_length=128,
                                                    truncation=True, return_tensors="pt")
                        tokenized_label = tokenized_label.data["input_ids"]
                        pad_tensor = torch.full(tokenized_label.shape, -100)
                        tokenized_label = torch.where(tokenized_label != 0, tokenized_label, pad_tensor)

                        sample.pop("input")
                        sample.pop("alpaca_input")
                        sample.pop("chatglm_input")
                        sample["input_ids"] = tokenized_input["input_ids"]
                        sample["attention_mask"] = tokenized_input["attention_mask"]
                        sample["alpaca_input_ids"] = tokenized_alpaca_input["input_ids"]
                        sample["alpaca_attention_mask"] = tokenized_alpaca_input["attention_mask"]
                        sample["chatglm_input_ids"] = tokenized_chatglm_input["input_ids"]
                        sample["chatglm_attention_mask"] = tokenized_chatglm_input["attention_mask"]
                        sample["label"] = tokenized_label
                        self.samples[task].append(sample)
                    elif task == "Recommend":
                        tokenized_input = tokenizer(separator.join(sample["input"]), padding="max_length",
                                                    truncation=True, max_length=512, return_tensors="pt",
                                                    return_attention_mask=True)
                        sample.pop("input")
                        sample["input_ids"] = tokenized_input["input_ids"]
                        sample["attention_mask"] = tokenized_input["attention_mask"]
                        self.samples[task].append(sample)
                    else:
                        print("Evaluate or other task is not available.")
                        assert False

        save_dir = os.path.join(self.args.data_root_path, "saved_data")
        pickle.dump(self.samples, open(os.path.join(save_dir, f"{self.mode}_samples.pkl"), "wb"))
        if not os.path.exists(os.path.join(save_dir, "jsonl")):
            os.makedirs(os.path.join(save_dir, "jsonl"))
        with open(os.path.join(save_dir, "jsonl", f"{self.mode}_sample.jsonl"), "w", encoding="UTF-8") as f:
            lines = [json.dumps(sample, ensure_ascii=False) for sample in self.text_samples]
            f.write("\n".join(lines))
        if not os.path.exists(os.path.join(save_dir, "json")):
            os.makedirs(os.path.join(save_dir, "json"))
        with open(os.path.join(save_dir, "json", f"{self.mode}_sample.json"), "w", encoding="UTF-8") as f:
            json.dump(self.text_samples, f, ensure_ascii=False, indent=2)


class UNeed(Dataset):
    def __init__(self, args, mode, task=None):
        data_process = DataProcess(args, mode)
        samples = data_process.check_and_process()
        if task is None:
            self.samples = [sample for temp in samples.values() for sample in temp]
        else:
            self.samples = samples[task]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    return list(data)
