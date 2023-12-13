# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
import transformers
from transformers import BertModel
# from torch.utils.data import *
from copy import deepcopy

from base_models.module import Embeddings, Encoder, LayerNorm
# from .module import *


class SASRecModel(nn.Module):
    def __init__(self, args, embeddings=None):
        super(SASRecModel, self).__init__()
        if embeddings is None:
            self.embeddings = Embeddings(args)
        else:
            self.embeddings = embeddings
        self.encoder = Encoder(args)
        self.args = deepcopy(args)

        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

        self.apply(self.init_sas_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (bs, seq_len)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2)  # torch.int64, (bs, 1, 1, seq_len)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape),
                                     diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.to(torch.device("cuda:0"))
        extended_attention_mask = extended_attention_mask * subsequent_mask

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding = self.embeddings(input_ids)

        encoded_layers = self.encoder(
            embedding,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        # [B L H]
        sequence_output = encoded_layers[-1]
        return sequence_output

    def init_sas_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)
        self.to(self.args.device)

    def load_model(self, path):
        load_states = torch.load(path, map_location=self.args.device)
        load_states_keys = set(load_states.keys())
        this_states_keys = set(self.state_dict().keys())
        assert this_states_keys.issubset(this_states_keys)
        key_not_used = load_states_keys - this_states_keys
        for key in key_not_used:
            del load_states[key]

        self.load_state_dict(load_states)

    def compute_loss(self, y_pred, y, subset='test'):
        pass

    def cross_entropy(self, seq_out, pos_ids, neg_ids, use_cuda=True):

        # [batch seq_len hidden_size]
        pos_emb = self.embeddings.item_embeddings(pos_ids)
        neg_emb = self.embeddings.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        # [batch*seq_len hidden_size]
        seq_emb = seq_out.view(-1, self.args.hidden_size)

        # [batch*seq_len]
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)

        # [batch*seq_len]
        istarget = (pos_ids > 0).view(
            pos_ids.size(0) * self.args.max_seq_length).float()
        loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) *
                         istarget -
                         torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) *
                         istarget) / torch.sum(istarget)

        return loss


class BERTModel(nn.Module):
    def __init__(self, args, num_class, bert_embed_size=768):
        super(BERTModel, self).__init__()
        bert_path = args.bert_path
        init_add = args.init_add
        self.args = args

        self.bert = BertModel.from_pretrained(args.bert_path)
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        # self.bert = AlbertModel.from_pretrained("clue/albert_chinese_tiny")
        # 312

        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(bert_embed_size, num_class)
        self.add_name = 'addition_model.pth'
        if init_add:
            self.load_addition_params(join(bert_path, self.add_name))

    def forward(self, x, raw_return=True):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[
            2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # _, pooled = self.bert(context,
        #                       token_type_ids=types,
        #                       attention_mask=mask)

        pooled = self.bert(context,
                           token_type_ids=types,
                           attention_mask=mask)

        if raw_return:
            return pooled.pooler_output
        else:
            return self.fc(pooled.pooler_output)

    def compute_loss(self, y_pred, y, subset='test'):
        # ipdb.set_trace()
        loss = F.cross_entropy(y_pred, y)

        return loss

    def save_model(self, save_path):
        self.bert.save_pretrained(save_path)
        torch.save(self.fc.state_dict(), join(save_path, self.add_name))

    def load_addition_params(self, path):
        self.fc.load_state_dict(torch.load(path,
                                           map_location=self.args.device))


class Fusion(nn.Module):
    def __init__(self, sasrec_size, bart_size, hidden_size):
        super(Fusion, self).__init__()
        concat_embed_size = bart_size + sasrec_size
        self.fc = nn.Linear(concat_embed_size, hidden_size)

    def forward(self, SASRec_out, BERT_out):
        representation = torch.cat((SASRec_out, BERT_out), dim=1)
        representation = self.fc(representation)

        return representation

    # def save_model(self, file_name):
    #     torch.save(self.cpu().state_dict(), file_name)
    #     # self.to(self.args.device)

    # def load_model(self, path):
    #     self.load_state_dict(torch.load(path)


class SASBERT(nn.Module):
    def __init__(self, opt, args, num_class, bert_embed_size=768):
        super(SASBERT, self).__init__()
        self.args = args
        self.opt = opt

        # bert
        self.BERT = BERTModel(self.args, num_class)
        self.SASREC = SASRecModel(self.args)
        self.fusion = Fusion(args, num_class)

        if args.load_model:
            self.SASREC.load_model(self.args.sasrec_load_path)
            self.fusion.load_model(args.fusion_load_path)

    def forward(self, x):
        x_bert = x[:3]
        pooled = self.BERT(x_bert, raw_return=True)  # bs, hidden_size1

        input_ids, target_pos, input_mask, sample_negs = x[-4:]
        sequence_output = self.SASREC(
            input_ids, input_mask,
            self.args.use_cuda)  # bs, max_len, hidden_size2
        sequence_output = sequence_output[:, -1, :]  # bs, hidden_size2

        representation = self.fusion(sequence_output, pooled)  # bs, num_class

        return representation

    def save_model(self, module_name):
        if 'BERT' in module_name:
            self.BERT.save_model(self.opt['model_save_path'])
        if 'SASRec' in module_name:
            self.SASREC.save_model(self.opt['sasrec_save_path'])
        if 'Fusion' in module_name:
            self.fusion.save_model(self.opt['fusion_save_path'])

    def load_model(self, module_name, path):
        pass

    def compute_loss(self, y_pred, y, subset='test'):
        loss = F.cross_entropy(y_pred, y)
        return loss

    def get_optimizer(self):
        bert_param_optimizer = list(self.BERT.named_parameters())  # 模型参数名字列表
        # bert_param_optimizer = [p for n, p in bert_param_optimizer]
        other_param_optimizer = list(self.SASREC.named_parameters()) + \
                                list(self.fusion.named_parameters())
        other_param_optimizer = [p for n, p in other_param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt['lr_bert'])
        optimizer = transformers.AdamW(
            [
                # {'params': optimizer_grouped_parameters, 'lr': self.opt['lr_bert']},
                {
                    'params': [
                        p for n, p in bert_param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    'lr':
                        self.opt['lr_bert']
                },
                {
                    'params': [
                        p for n, p in bert_param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    'lr':
                        self.opt['lr_bert']
                },
                {
                    'params': other_param_optimizer
                }
            ],
            lr=self.opt['lr_sasrec'])

        return optimizer
