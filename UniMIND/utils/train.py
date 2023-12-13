import os.path

import torch
import wandb
from pytorch_transformers import WarmupLinearSchedule
from tqdm import tqdm
from transformers import AdamW
import random
from available_resources.datasets.u_need.u_need import UNeed, collate_fn
from utils.eval import evaluate_metric
from torch.utils.data import DataLoader


def finetune_model(args, model, train_dataloader, valid_dataloader, task):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.finetune_learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) * args.finetune_epoch_num
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    best_valid_score = 0

    save_model_dir = os.path.join(args.save_model_path, task.lower() + "_finetune")
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    for epoch in range(args.finetune_epoch_num):
        train(args, model, train_dataloader, optimizer, scheduler)

        valid_dataset = UNeed(args, "valid", task)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_fn)
        output_dir = os.path.join(args.save_result_path, task.lower() + "_finetune", f"{epoch}_epoch", "valid")
        valid_result = evaluate_metric(args, model, valid_dataloader, "valid", task, output_dir,
                                       output_eval_data=True, output_crs_data=False)

        valid_score = valid_result["所有行业"][args.choose_metric[task]]

        if not args.no_wandb:
            wandb.log({f"valid_{task.lower()}_score": valid_score})

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            save_model_path = os.path.join(save_model_dir, "best_valid.pt")
            torch.save(model.state_dict(), save_model_path)

        test_dataset = UNeed(args, "test", task)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=collate_fn)
        output_dir = os.path.join(args.save_result_path, task.lower() + "_finetune", f"{epoch}_epoch", "test")
        evaluate_metric(args, model, test_dataloader, "test", task, output_dir, output_eval_data=True,
                        output_crs_data=False)

        if epoch % 5 == 0:
            save_model_path = os.path.join(save_model_dir, f"{epoch}_epoch.pt")
            torch.save(model.state_dict(), save_model_path)

    return os.path.join(save_model_dir, "best_valid.pt")


def train_model(args, model, train_dataloader, valid_dataloader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.train_learning_rate, eps=args.adam_epsilon)
    t_total = len(train_dataloader) * args.train_epoch_num
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    best_valid_score = 0
    save_model_dir = os.path.join(args.save_model_path, "all_tasks")
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    for epoch in range(args.train_epoch_num):
        train(args, model, train_dataloader, optimizer, scheduler)

        valid_score = 0
        for task in ["Understand", "Elicit", "Recommend", "Response"]:
            valid_dataset = UNeed(args, "valid", task)
            valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
                                          collate_fn=collate_fn)
            output_dir = os.path.join(args.save_result_path, "all_tasks", f"{epoch}_epoch", "valid")
            valid_result = evaluate_metric(args, model, valid_dataloader, "valid", task, output_dir,
                                           output_eval_data=True, output_crs_data=False)
            score = valid_result["所有行业"][args.choose_metric[task]]

            if not args.no_wandb:
                wandb.log({f"valid_{task.lower()}_score": score})

            valid_score += score

            test_dataset = UNeed(args, "test", task)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=collate_fn)
            output_dir = os.path.join(args.save_result_path, "all_tasks", f"{epoch}_epoch", "test")
            evaluate_metric(args, model, test_dataloader, "test", task, output_dir, output_eval_data=True,
                            output_crs_data=False)

        if not args.no_wandb:
            wandb.log({"valid_total_score": valid_score})

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            save_model_path = os.path.join(save_model_dir, "best_valid.pt")
            torch.save(model.state_dict(), save_model_path)

        if epoch % 5 == 0:
            save_model_path = os.path.join(save_model_dir, f"{epoch}_epoch.pt")
            torch.save(model.state_dict(), save_model_path)

    return os.path.join(save_model_dir, "best_valid.pt")


def train(args, model, train_dataloader, optimizer, scheduler):
    model.train()
    model.zero_grad()

    accumulative_loss = 0.0
    total_loss = 0.0
    for batch_index, samples in enumerate(tqdm(train_dataloader)):
        if args.sample_random and random.randint(0, 30) != 3:
            continue
        if args.llm == "none":
            llm_mode = ""
        else:
            llm_mode = args.llm + "_"

        device = torch.device("cuda:0")

        # sample中的字段
        # sample = {  
        #     "input_ids": ,
        #     "attention_mask": ,
        #     "domain": ,
        #     "user_id": ,
        #     "sid": ,
        #     "seller_id": ,
        #     "task_name": ,
        #     "sample_index_in_dialogue": ,
        #     "sample_id": ,
        #     "alpaca_input_ids": ,  Only task 1,24
        #     "alpaca_attention_mask": , Only task 1,24
        #     "chatgpt_input_ids": , Only task 1,24
        #     "chatgpt_attention_mask": , Only task 1,24
        #     "label": , Only task 1,24
        #     "item_id": , Only task 3
        #     "alpaca_item_id": , Only task 3
        #     "chatglm_item_id": , Only task 3 
        #     "candidates": , Only task 3
        # }

        input_ids = []
        attention_mask = []
        labels = []
        for sample in samples:
            if sample["task_name"] != "Recommend":
                input_ids.append(sample[llm_mode + "input_ids"])
                attention_mask.append(sample[llm_mode + "attention_mask"])
                labels.append(sample["label"])
        if len(input_ids) != 0:
            input_ids = torch.cat(input_ids, dim=0).to(device)
            attention_mask = torch.cat(attention_mask, dim=0).to(device)
            labels = torch.cat(labels, dim=0).to(device)
            generate_loss = model(input_ids, attention_mask, labels=labels)
        else:
            generate_loss = 0

        input_ids = []
        attention_mask = []
        item_ids = []
        llm_item_ids = []
        sids = []
        for sample in samples:
            if sample["task_name"] == "Recommend":
                input_ids.append(sample["input_ids"])
                attention_mask.append(sample["attention_mask"])
                item_ids.append(sample["item_id"])
                llm_item_ids.append(sample[llm_mode + "item_id"])
                sids.append(sample["sid"])
        if len(input_ids) != 0:
            input_ids = torch.cat(input_ids, dim=0).to(device)
            attention_mask = torch.cat(attention_mask, dim=0).to(device)
            recommend_loss = model(input_ids, attention_mask, item_ids=item_ids, llm_item_ids=llm_item_ids, sids=sids)
        else:
            recommend_loss = 0

        train_loss = generate_loss + recommend_loss

        if not args.no_wandb:
            wandb.log({"train_loss": train_loss, "train_generate_loss": generate_loss,
                       "train_recommend_loss": recommend_loss})

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        accumulative_loss += train_loss
        total_loss += train_loss
        model.zero_grad()
        torch.cuda.empty_cache()

        if batch_index % args.log_step_num == 0:
            print(accumulative_loss / args.log_step_num)
            accumulative_loss = 0
    print("Train total loss: " + str(total_loss / len(train_dataloader)))
