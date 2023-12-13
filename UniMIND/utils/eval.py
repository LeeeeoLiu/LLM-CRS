import json
import os
import random

import torch
from tqdm import tqdm
import debugpy

# debugpy.listen(("0.0.0.0", 15678))

import wandb
from transformers import BertTokenizer

from available_resources.metrics.acc import domain_acc_evaluate
from available_resources.metrics.gen import domain_gen_evaluate
from available_resources.metrics.rec import domain_recommend_evaluate


# def evaluate_loss(args, model, valid_dataloader):
#     model.eval()

#     with torch.no_grad():
#         total_loss = 0.0
#         accumulative_loss = 0.0
#         for batch_index, samples in enumerate(tqdm(valid_dataloader)):
#             if args.sample_random and random.randint(0, 20) != 5:
#                 continue
#             if args.llm == "none":
#                 llm_mode = ""
#             else:
#                 llm_mode = args.llm+"_"

#             device = torch.device("cuda:0")

#             input_ids = []
#             attention_mask = []
#             labels = []
#             for sample in samples:
#                 if sample["task_name"] != "Recommend":
#                     input_ids.append(sample[llm_mode+"input_ids"])
#                     attention_mask.append(sample[llm_mode+"attention_mask"])
#                     labels.append(sample["label"])
#             if len(input_ids) != 0:
#                 input_ids = torch.cat(input_ids, dim=0).to(device)
#                 attention_mask = torch.cat(attention_mask, dim=0).to(device)
#                 labels = torch.cat(labels, dim=0).to(device)
#                 generate_loss = model(input_ids, attention_mask, labels=labels)
#             else:
#                 generate_loss = 0

#             input_ids = []
#             attention_mask = []
#             item_ids = []
#             llm_item_ids = []
#             sids = []
#             for sample in samples:
#                 if sample["task_name"] == "Recommend":
#                     input_ids.append(sample["input_ids"])
#                     attention_mask.append(sample["attention_mask"])
#                     item_ids.append(sample["item_id"])
#                     llm_item_ids.append(sample[llm_mode+"item_id"])
#                     sids.append(sample["sid"])
#             if len(input_ids) != 0:
#                 input_ids = torch.cat(input_ids, dim=0).to(device)
#                 attention_mask = torch.cat(attention_mask, dim=0).to(device)
#                 recommend_loss = model(input_ids, attention_mask, item_ids=item_ids,llm_item_ids=llm_item_ids, sids=sids)
#             else:
#                 recommend_loss = 0

#             valid_loss = generate_loss + recommend_loss

#             if not args.no_wandb:
#                 wandb.log({"valid_loss": valid_loss, "valid_generate_loss": generate_loss,
#                            "valid_recommend_loss": recommend_loss})

#             accumulative_loss += valid_loss
#             total_loss += valid_loss

#             if batch_index % args.log_step_num == 0:
#                 print(accumulative_loss / args.log_step_num)
#                 accumulative_loss = 0
#         print("Valid total loss: " + str(total_loss / len(valid_dataloader)))
#     return total_loss / len(valid_dataloader)


def evaluate_metric(args, model, dataloader, mode, task, output_dir, output_eval_data=True, output_crs_data=False):
    tokenizer = BertTokenizer.from_pretrained(args.base_model_path, additional_special_tokens=args.special_tokens)
    model.eval()
    with torch.no_grad():
        output = []
        if args.llm == "none":
            llm_mode = ""
        else:
            llm_mode = args.llm + "_"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if output_crs_data:
            merge_dump_filename = os.path.join(output_dir, f"merge_{mode}_samples.json")

            if not os.path.exists(merge_dump_filename):
                temp_path = os.path.join(args.data_root_path, "prompt4LLM", f"{mode}_sample.json")
                merge_data = json.load(open(temp_path, "r", encoding="UTF-8"))
            else:
                merge_data = json.load(open(merge_dump_filename, "r", encoding="UTF-8"))

            merge_data_dict = {sample["sample_id"]: sample for sample in merge_data}

        if task in ["Understand", "Elicit", "Response"]:
            domain_data = []
            for _, samples in enumerate(tqdm(dataloader)):
                if args.sample_random and random.randint(0, 20) != 3:
                    continue

                domain_data.extend([sample["domain"] for sample in samples])

                device = torch.device("cuda:0")
                input_ids = torch.cat([sample[llm_mode + "input_ids"] for sample in samples], dim=0).to(device)
                attention_mask = torch.cat([sample[llm_mode + "attention_mask"] for sample in samples], dim=0).to(
                    device)
                labels = torch.cat([sample["label"] for sample in samples], dim=0).to(device)

                generate_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                              num_beams=args.beam_num, max_length=args.max_response_length,
                                              early_stopping=True)

                pad_tensor = torch.full(labels.shape, 0).to(torch.device("cuda", 0))
                labels = torch.where(labels != -100, labels, pad_tensor)
                batch_output = []
                for i in range(generate_ids.shape[0]):
                    batch_output.append([
                        tokenizer.decode(input_ids[i][:int(torch.sum(attention_mask[i]))],
                                         skip_special_tokens=False).replace(" ", ""),
                        tokenizer.decode(generate_ids[i], skip_special_tokens=True).replace(" ", ""),
                        tokenizer.decode(labels[i], skip_special_tokens=True).replace(" ", "")
                    ])
                output.extend(batch_output)
                
                if output_crs_data:
                    for i, sample in enumerate(samples):
                        merge_data_dict[sample["sample_id"]][f"{llm_mode}{args.base_name.replace('-', '_')}_crs_output"] = \
                        batch_output[i][1]

            result = {}
            if output_eval_data:
                if task == "Understand":
                    predict = [set([tuple(i.split(":")) for i in p.split("；")]) for _, p, _ in output]
                    golden = [set([tuple(i.split(":")) for i in g.split("；")]) for _, _, g in output]
                    result = domain_acc_evaluate(predict, golden, domain_data)
                elif task == "Elicit":
                    predict = [set(p.split("；")) for _, p, _ in output]
                    golden = [set(g.split("；")) for _, _, g in output]
                    result = domain_acc_evaluate(predict, golden, domain_data)
                elif task == "Response":
                    predict = [p for _, p, _ in output]
                    golden = [g for _, _, g in output]
                    result = domain_gen_evaluate(predict, golden, domain_data)
                else:
                    assert False

                output_filename = os.path.join(output_dir, task + ".txt")
                with open(output_filename, "w", encoding="UTF-8") as f:
                    for d, r in result.items():
                        f.write(d + ":\n")
                        f.write("\n".join([f"{metric}:{value}" for metric, value in r.items()]))
                        f.write("\n\n")
                    f.write("\n".join([f"Input: {i[0]}\nGenerate: {i[1]}\nGolden: {i[2]}\n" for i in output]))

            if output_crs_data:
                merge_data = list(merge_data_dict.values())
                json.dump(merge_data, open(merge_dump_filename, "w", encoding="UTF-8"), ensure_ascii=False, indent=2)

            return result

        elif task == "Recommend":
            device = torch.device("cuda:0")

            domain_data = []
            item_id_ranks = []
            ground_item_ids = []
            for _, samples in enumerate(tqdm(dataloader)):
                if args.sample_random and random.randint(0, 20) != 3:
                    continue

                domain_data.extend([sample["domain"] for sample in samples])

                input_ids = torch.cat([sample["input_ids"] for sample in samples], dim=0).to(device)
                attention_mask = torch.cat([sample["attention_mask"] for sample in samples], dim=0).to(device)
                candidates = [sample["candidate"] for sample in samples]

                if llm_mode != "":
                    llm_item_ids = [sample[llm_mode + "item_id"] for sample in samples]
                else:
                    llm_item_ids = None

                sids = [sample["sid"] for sample in samples]
                rank = model.recommend_item(input_ids, attention_mask, candidates, llm_item_ids, sids)
                item_id_ranks.extend(rank)

                ground_item_ids.extend([sample["item_id"] for sample in samples])

                if output_crs_data:
                    for i, sample in enumerate(samples):
                        merge_data_dict[sample["sample_id"]][f"{llm_mode}{args.base_name.replace('-', '_')}_crs_output"] = rank[i]

            result = {}
            if output_eval_data:
                result = domain_recommend_evaluate(item_id_ranks, ground_item_ids, [1, 5, 10, 15], domain_data)
                output_filename = os.path.join(output_dir, task + ".txt")

                with open(output_filename, "w", encoding="UTF-8") as f:
                    for d, r in result.items():
                        f.write(d + ":\n")
                        f.write("\n".join([f"{metric}:{value}" for metric, value in r.items()]))
                        f.write("\n\n")

            if output_crs_data:
                merge_data = list(merge_data_dict.values())
                json.dump(merge_data, open(merge_dump_filename, "w", encoding="UTF-8"), ensure_ascii=False, indent=2)

            return result
        else:
            assert False
