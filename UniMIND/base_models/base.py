import json
import os.path
import pickle

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BartForConditionalGeneration, BartForSequenceClassification
from .modeling_cpt import CPTForConditionalGeneration
from base_models.sasrec import Fusion, SASRecModel


# import debugpy
# debugpy.listen(("0.0.0.0", 15678))


class BARTCRSModel(nn.Module):
    def __init__(self, args):
        super(BARTCRSModel, self).__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(args.base_model_path)
        self.bart.resize_token_embeddings(args.vocab_size)

        # Recommend
        path = os.path.join(args.data_root_path, "saved_data", "item_id2index.pkl")
        self.item_id2index = pickle.load(open(path, "rb"))
        self.sid2seq = json.load(open(os.path.join(args.data_root_path, "allsid2seq.json")))
        self.item_num = args.item_num
        d_model = self.bart.config.d_model

        self.item_embedding = nn.Embedding(num_embeddings=self.item_num, embedding_dim=300)
        self.item_bias = nn.Linear(in_features=300, out_features=self.item_num)
        self.recommend_criterion = nn.CrossEntropyLoss()

        self.SASREC = SASRecModel(args, self.item_embedding)
        self.fusion = Fusion(sasrec_size=args.hidden_size, bart_size=d_model, hidden_size=300)

        layers = [self.fusion, self.item_embedding, self.item_bias]

        for layer in layers:
            for name, param in layer.named_parameters():
                if param.data.dim() > 1:
                    xavier_uniform_(param.data)

        # load item embedding from word2vec
        item_embedding_path = os.path.join(args.data_root_path, "item2vec-300d.pkl")
        item_embedding_dict = pickle.load(open(item_embedding_path, "rb"))
        for item_id in item_embedding_dict.keys():
            if item_id not in self.item_id2index.keys():
                continue
            tmp_item_embedding = torch.tensor(item_embedding_dict[item_id])
            self.item_embedding.weight.data[self.item_id2index[item_id]].copy_(tmp_item_embedding)

    def forward(self, input_ids, attention_mask, labels=None, item_ids=None, llm_item_ids=None, sids=None):
        if item_ids is None:
            output = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return output.loss
        else:
            item_indexes = [self.item_id2index[item_id] for item_id in item_ids]
            if llm_item_ids is not None:
                llm_item_indexes = [self.item_id2index[item_id] for item_id in llm_item_ids]
            else:
                llm_item_indexes = []

            with torch.no_grad():
                output = self.bart(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            eos_mask = input_ids.eq(102)
            hidden_state = output.decoder_hidden_states[-1][eos_mask, :]
            user_representation = self.fusion_historical_behaviors(sids, hidden_state)

            if llm_item_indexes:
                user_representation = user_representation + self.item_embedding.weight[llm_item_indexes,]
            item_embedding = self.item_embedding.weight.transpose(0, 1)
            score = torch.matmul(user_representation, item_embedding) + self.item_bias.bias
            recommend_loss = self.recommend_criterion(score, torch.tensor(item_indexes).to(torch.device("cuda:0")))
            return recommend_loss

    def fusion_historical_behaviors(self, sids, hidden_state):
        historical_actions = [self.sid2seq[sid] for sid in sids]
        max_len = max([len(historical_action) for historical_action in historical_actions])
        max_len = max(max_len, 1)
        historical_action_ids = []
        for historical_action in historical_actions:
            historical_action_ids.append(
                [self.item_id2index[item_id] for item_id in historical_action if item_id in self.item_id2index.keys()])

        his_ids = torch.tensor([historical_action_id + [0] * (max_len - len(historical_action_id))
                                for historical_action_id in historical_action_ids]).to(torch.device("cuda:0")).long()
        his_mask = torch.tensor([[1] * len(historical_action_id) + [0] * (max_len - len(historical_action_id))
                                 for historical_action_id in historical_action_ids]).to(torch.device("cuda:0"))

        # bs, max_len, hidden_size2
        sequence_output = self.SASREC(his_ids, his_mask)
        sequence_output = sequence_output[:, -1, :]  # bs, hidden_size2
        user_representation = self.fusion(sequence_output, hidden_state)
        return user_representation

    def recommend_item(self, input_ids, attention_mask, candidates, llm_item_ids=None, sids=None):
        # batch_size
        if llm_item_ids is not None:
            llm_item_indexes = [self.item_id2index[item_id] for item_id in llm_item_ids]
        else:
            llm_item_indexes = []

        # batch_size, 20
        candidates_index = [[self.item_id2index[item_id] for item_id in candidate] for candidate in candidates]

        output = self.bart(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # batch_size, 768
        eos_mask = input_ids.eq(102)
        hidden_state = output.decoder_hidden_states[-1][eos_mask, :]
        # batch_size, 300
        user_representation = self.fusion_historical_behaviors(sids, hidden_state)

        if llm_item_ids is not None:
            user_representation = user_representation + self.item_embedding.weight[llm_item_indexes,]

        ranks = []
        for i, candidate_index in enumerate(candidates_index):
            candidate_embedding = self.item_embedding.weight[candidate_index,].transpose(0, 1)
            score = torch.matmul(user_representation[i], candidate_embedding) + self.item_bias.bias[candidate_index,]
            rank = sorted(candidates[i], key=lambda x: score[candidates[i].index(x)], reverse=True)
            ranks.append(rank)

        return ranks

    def generate(self, **kwargs):
        return self.bart.generate(**kwargs)


class CPTCRSModel(nn.Module):
    def __init__(self, args):
        super(CPTCRSModel, self).__init__()
        self.cpt = CPTForConditionalGeneration.from_pretrained(args.base_model_path)
        self.cpt.resize_token_embeddings(args.vocab_size)

        # Recommend
        path = os.path.join(args.data_root_path, "saved_data", "item_id2index.pkl")
        self.item_id2index = pickle.load(open(path, "rb"))
        self.sid2seq = json.load(open(os.path.join(args.data_root_path, "allsid2seq.json")))
        self.item_num = args.item_num
        d_model = self.cpt.config.d_model * 2

        self.item_embedding = nn.Embedding(num_embeddings=self.item_num, embedding_dim=300)
        self.item_bias = nn.Linear(in_features=300, out_features=self.item_num)
        self.recommend_criterion = nn.CrossEntropyLoss()

        self.SASREC = SASRecModel(args, self.item_embedding)
        self.fusion = Fusion(sasrec_size=args.hidden_size, bart_size=d_model, hidden_size=300)

        layers = [self.fusion, self.item_embedding, self.item_bias]

        for layer in layers:
            for name, param in layer.named_parameters():
                if param.data.dim() > 1:
                    xavier_uniform_(param.data)

        # load item embedding from word2vec
        item_embedding_path = os.path.join(args.data_root_path, "item2vec-300d.pkl")
        item_embedding_dict = pickle.load(open(item_embedding_path, "rb"))
        for item_id in item_embedding_dict.keys():
            if item_id not in self.item_id2index.keys():
                continue
            tmp_item_embedding = torch.tensor(item_embedding_dict[item_id])
            self.item_embedding.weight.data[self.item_id2index[item_id]].copy_(tmp_item_embedding)

    def forward(self, input_ids, attention_mask, labels=None, item_ids=None, llm_item_ids=None, sids=None):
        if item_ids is None:
            output = self.cpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return output.loss
        else:
            item_indexes = [self.item_id2index[item_id] for item_id in item_ids]
            if llm_item_ids is not None:
                llm_item_indexes = [self.item_id2index[item_id] for item_id in llm_item_ids]
            else:
                llm_item_indexes = []

            with torch.no_grad():
                output = self.cpt.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            enc_hidden_state = output.encoder_last_hidden_state[:, 0, :]
            eos_mask = input_ids.eq(102)
            dec_hidden_state = output.last_hidden_state[eos_mask, :]
            hidden_state = torch.cat([enc_hidden_state, dec_hidden_state], dim=-1)
            user_representation = self.fusion_historical_behaviors(sids, hidden_state)

            if llm_item_indexes:
                user_representation = user_representation + self.item_embedding.weight[llm_item_indexes,]
            item_embedding = self.item_embedding.weight.transpose(0, 1)
            score = torch.matmul(user_representation, item_embedding) + self.item_bias.bias
            recommend_loss = self.recommend_criterion(score, torch.tensor(item_indexes).to(torch.device("cuda:0")))
            return recommend_loss

    def fusion_historical_behaviors(self, sids, hidden_state):
        historical_actions = [self.sid2seq[sid] for sid in sids]
        max_len = max([len(historical_action)
                       for historical_action in historical_actions])
        max_len = max(max_len, 1)
        historical_action_ids = []
        for historical_action in historical_actions:
            historical_action_ids.append(
                [self.item_id2index[item_id] for item_id in historical_action if item_id in self.item_id2index.keys()])

        his_ids = torch.tensor([historical_action_id + [0] * (max_len - len(historical_action_id))
                                for historical_action_id in historical_action_ids]).to(torch.device("cuda:0")).long()
        his_mask = torch.tensor([[1] * len(historical_action_id) + [0] * (max_len - len(historical_action_id))
                                 for historical_action_id in historical_action_ids]).to(torch.device("cuda:0"))

        # bs, max_len, hidden_size2
        sequence_output = self.SASREC(his_ids, his_mask)
        sequence_output = sequence_output[:, -1, :]  # bs, hidden_size2
        user_representation = self.fusion(sequence_output, hidden_state)
        return user_representation

    def recommend_item(self, input_ids, attention_mask, candidates, llm_item_ids=None, sids=None):
        # batch_size
        if llm_item_ids is not None:
            llm_item_indexes = [self.item_id2index[item_id] for item_id in llm_item_ids]
        else:
            llm_item_indexes = []

        # batch_size, 20
        candidates_index = [[self.item_id2index[item_id] for item_id in candidate] for candidate in candidates]

        output = self.cpt.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        enc_hidden_state = output.encoder_last_hidden_state[:, 0, :]
        eos_mask = input_ids.eq(102)
        dec_hidden_state = output.last_hidden_state[eos_mask, :]
        hidden_state = torch.cat([enc_hidden_state, dec_hidden_state], dim=-1)
        user_representation = self.fusion_historical_behaviors(sids, hidden_state)

        if llm_item_ids is not None:
            user_representation = user_representation + self.item_embedding.weight[llm_item_indexes,]

        ranks = []
        for i, candidate_index in enumerate(candidates_index):
            candidate_embedding = self.item_embedding.weight[candidate_index,].transpose(0, 1)
            score = torch.matmul(user_representation[i], candidate_embedding) + self.item_bias.bias[candidate_index,]
            rank = sorted(candidates[i], key=lambda x: score[candidates[i].index(x)], reverse=True)
            ranks.append(rank)

        return ranks

    def generate(self, **kwargs):
        return self.cpt.generate(**kwargs)
