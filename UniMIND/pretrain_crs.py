import argparse
import os.path
import time
import torch
import wandb
from torch.utils.data import DataLoader

from available_resources.datasets.u_need.u_need import UNeed, collate_fn
from utils.eval import evaluate_metric
from base_models.base import BARTCRSModel, CPTCRSModel
from utils.train import finetune_model, train_model

parser = argparse.ArgumentParser(description='An unified conversational recommend system based on U-Need Dataset')

parser.add_argument("--sample_random", action="store_true", default=False, help="use part of data(default=false)")
parser.add_argument("--no_wandb", action="store_true", default=False, help="whether use wandb(default=true)")
parser.add_argument("--test", action="store_true", default=False, help="whether test(default=false)")
parser.add_argument("--base_name", type=str, default="bart-base",
                    help="crs base model(available: bart-base, bart-large, cpt-base, cpt-large)")

parser.add_argument('--log_step_num', type=int, default=30, help="print loss after how many steps")
parser.add_argument("--max_context_length", default=1024, type=int, help="max context length")
parser.add_argument("--max_response_length", default=128, type=int, help="max response length")

parser.add_argument("--train_epoch_num", type=int, default=15, help="train procedure epoch num")
parser.add_argument("--train_learning_rate", type=float, default=1e-5, help="initial learning rate for Adam")
parser.add_argument("--finetune_epoch_num", type=int, default=15, help="finetune procedure epoch num")
parser.add_argument("--finetune_learning_rate", type=float, default=1e-3, help="initial learning rate for Adam")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")

parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup steps")
parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon for Adam")
parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm")
parser.add_argument('--beam_num', type=int, default=3, help="beam num")
parser.add_argument('--reward', type=str, default="none", help="reward type")
parser.add_argument('--llm', type=str, default="none", help="llm name, available: alpaca, chatglm. Default not use llm")
parser.add_argument("--hidden_size", type=int, default=300, help="hidden size of transformer model")
parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
parser.add_argument('--num_attention_heads', default=1, type=int)
parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2, help="attention dropout p")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="hidden dropout p")
parser.add_argument("--initializer_range", type=float, default=0.02)
parser.add_argument('--max_seq_length', default=100, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    args.item_num = 68954
    args.vocab_size = 51278
    args.recommend_finetune = True
    args.begin_time = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    args.data_root_path = "./available_resources/datasets/u_need"
    if args.base_name == "bart-base":
        args.base_model_path = "./available_resources/models/bart-base-chinese"
    elif args.base_name == "bart-large":
        args.base_model_path = "./available_resources/models/bart-large-chinese"
    elif args.base_name == "cpt-base":
        args.base_model_path = "./available_resources/models/cpt-base"
    elif args.base_name == "cpt-large":
        args.base_model_path = "./available_resources/models/cpt-large"
    else:
        print("available base model: bart-base, bart-large, cpt-base, cpt-large")
        assert False

    args.save_model_path = "./saved_model/" + args.begin_time + "_" + args.base_name.replace("-", "_")
    args.save_result_path = "./experiments/generate_result/" + args.begin_time + "_" + args.base_name.replace("-", "_")

    args.choose_metric = {"Understand": "F1", "Elicit": "F1", "Recommend": "Hit@1", "Response": "Bleu@2"}
    args.special_tokens = ['[user]', '[system]', '[understand]', '[elicit]', '[recommend]', "[eval]", "[LLM]"]

    # if you don't have saved_data/ in ./available_resources/datasets/u_need,
    # You may need to uncomment these four lines to generate data.
    # train_dataset = UNeed(args, "train")
    # valid_dataset = UNeed(args, "valid")
    # test_dataset = UNeed(args, "test")
    # exit(0)

    if not args.no_wandb:
        # os.environ["WANDB_MODE"] = "offline"
        wandb.init(project='UnifiedCRS', config=args.__dict__)

    if "bart" in args.base_name:
        model = BARTCRSModel(args)
    else:
        model = CPTCRSModel(args)

    if torch.cuda.is_available():
        model.cuda()

    if args.test:
        load_model_path = os.path.join("your save path", "best_valid.pt")
    else:
        # Recommend Finetune
        if args.recommend_finetune:
            train_dataset = UNeed(args, "train", "Recommend")
            valid_dataset = UNeed(args, "valid", "Recommend")
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate_fn)
            load_model_path = finetune_model(args, model, train_dataloader, valid_dataloader, "Recommend")
            model.load_state_dict(torch.load(load_model_path))

        # All tasks
        train_dataset = UNeed(args, "train")
        valid_dataset = UNeed(args, "valid")
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate_fn)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate_fn)
        load_model_path = train_model(args, model, train_dataloader, valid_dataloader)
    
    model.load_state_dict(torch.load(load_model_path))
    print("Load model successful.")
    output_dir = os.path.join(args.save_result_path, "eval")
    for mode in ["train","valid"]:
        for task in ["Understand", "Elicit", "Recommend", "Response"]:
            dataset = UNeed(args, mode, task)
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)            
            evaluate_metric(args, model, dataloader, mode, task, output_dir, output_eval_data=False, output_crs_data=True)
    
    mode = "test"
    for task in ["Understand", "Elicit", "Recommend", "Response"]:
        dataset = UNeed(args, mode, task)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)            
        evaluate_metric(args, model, dataloader, mode, task, output_dir, output_eval_data=True, output_crs_data=True)
