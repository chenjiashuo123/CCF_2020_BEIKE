# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
from torch import nn
import gc
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import tqdm
from sklearn.metrics import f1_score
from model_bert import BertForSequenceClassification, BertForSequenceClassification_last2embedding_cls
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from adv_train import FGM


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

logger = logging.getLogger(__name__)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


filemap = {
    'train': '/train_pse.csv',
    'test': '/test.csv',
    'validation': '/dev.csv'
}

def open_file(path, mode = 'train'):
    filename =path + filemap[mode]
    texts = []
    with open(filename, encoding='utf-8') as f:
        f_csv = csv.reader(f, delimiter='\t')
        for line in f_csv:
            texts.append(line)
    return texts


def read_data(texts, mode='train'):
    queries, replies, labels = [], [], []
    for text in tqdm.tqdm(texts, desc='loading ' + mode):
        queries.append(text[2])
        replies.append(text[3])
        if mode == 'test':
            labels.append(0)
        else:
            labels.append(int(text[4]))
    return queries, replies, labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, texts):
        self.tokenizer = BertTokenizer.from_pretrained("model/chinese-roberta-large-wwm/vocab.txt")
        self.queries, self.replies, self.labels = read_data(texts=texts, mode=mode)

    def __getitem__(self, index):
        token = _convert_to_transformer_inputs(self.queries[index], self.replies[index], self.tokenizer, max_sequence_length=100)
        label = self.labels[index]
        return [torch.tensor(token[0]), torch.tensor(token[1]), torch.tensor(token[2]), torch.tensor(label)]

    def __len__(self):
        return len(self.queries)


def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation=True,
                                       # truncation=True
                                       )

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, True, max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs)


def main():
    output_dir_o = './out_model/roberta_cls_pse_adv_2e-5/'
    train_batch_size = 16
    eval_batch_size = 128
    setup_seed(324)
    path = 'model/chinese-roberta-large-wwm/pytorch_model.bin'
    dir = './data/data_KFold/'
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = open_file('./data/data_KFold/data_origin_0', 'test')
    test_dataset = Dataset('test', test_data)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    try:
        os.makedirs(output_dir_o)
    except:
        pass
    config = BertConfig.from_pretrained('model/chinese-roberta-large-wwm/config.json')
    res_proba = np.zeros((len(test_dataset), 2))
    for i in range(5):
        name = 'data_origin_{}'.format(i)
        data_dir = dir + name
        output_dir = output_dir_o + 'bert_{}'.format(i)
        weight_decay = 0.01
        learning_rate = 2e-5
        nb_tr_examples, nb_tr_steps = 0, 0
        try:
            os.makedirs(output_dir)
        except:
            pass

        train_texts = open_file(data_dir, mode="train")
        train_data = Dataset('train', train_texts)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, shuffle=False,
                                      num_workers=0)
        epoch_steps = len(train_data) / train_batch_size
        total_steps = epoch_steps * epochs

        model = BertForSequenceClassification_last2embedding_cls(path=path, config=config)
        encoder_temp = getattr(model, 'bert')
        for layer in encoder_temp.encoder.layer[-5:]:
            for module in layer.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                elif isinstance(module, torch.nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
        fgm = FGM(model)
        model.to(device)
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer]
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

        global_step = 0
        if i != 0:
            logger.info("*" * 80)
            logger.info("*" * 80)
            logger.info("*" * 80)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", total_steps)
        logger.info("  Num Fold = %d", i)
        for epoch in range(epochs):
            loader_t = tqdm.tqdm(train_dataloader, desc='epoch:{}/{}'.format(epoch, epochs))
            for step, (batch_input_ids, batch_input_mask, segment_ids, batch_label) in enumerate(loader_t):
                batch_input_ids = batch_input_ids.to(device)
                batch_input_mask = batch_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                batch_label = batch_label.to(device)
                model.train()
                loss, _ = model(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                                token_type_ids=segment_ids, labels=batch_label)
                nb_tr_examples += batch_input_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                
                fgm.attack() 
                loss_adv = model(input_ids=batch_input_ids, attention_mask=batch_input_mask,
                                token_type_ids=segment_ids, labels=batch_label)[0]
                loss_adv.backward() 
                fgm.restore() 
                
                del batch_input_ids, batch_input_mask, batch_label
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loader_t.set_postfix(training="loss:{:.6f}".format(loss.item()))
                global_step += 1
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, "pytorch_model_{}.bin".format(i))
        torch.save(model_to_save.state_dict(), output_model_file)
        for file, flag in [('dev.csv', 'validation'), ('test.csv', 'test')]:
            inference_labels = []
            gold_labels = []
            eval_data = open_file(data_dir, flag)
            eval_dataset = Dataset(flag, eval_data)
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,
                                         shuffle=False,
                                         num_workers=0)
            model.eval()
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=input_mask,
                                 token_type_ids=segment_ids).detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                inference_labels.append(logits)
                gold_labels.append(label_ids)
            if flag == 'test':
                logits = np.concatenate(inference_labels, 0)
                res_proba += logits
                df = pd.read_csv(os.path.join(data_dir, file), encoding="utf-8", sep='\t', header=None)
                df.columns = ['query_id', 'reply_id', 'query', 'reply']
                df['logits_0'] = logits[:, 0]
                df['logits_1'] = logits[:, 1]
                df['label'] = np.argmax(logits, axis=1)
                df[['query_id', 'reply_id', 'label', 'logits_0', 'logits_1']].to_csv(
                    os.path.join(output_dir, "sub_{}.csv".format(i)), index=False,
                    sep="\t", header=None)
            if flag == 'validation':
                gold_labels = np.concatenate(gold_labels, 0)
                logits = np.concatenate(inference_labels, 0)
                print(flag, accuracy(logits, gold_labels))
                df = pd.read_csv(os.path.join(data_dir, file), encoding="utf-8", sep='\t', header=None)
                df.columns = ['query_id', 'reply_id', 'query', 'reply', 'label']
                df['pred'] = np.argmax(logits, axis=1)
                df['logits_0'] = logits[:, 0]
                df['logits_1'] = logits[:, 1]
                df[['query_id', 'reply_id', 'pred', 'label', 'logits_0', 'logits_1']].to_csv(
                    os.path.join(output_dir, "logits_dev_{}.csv".format(i)), index=False,
                    sep="\t", header=None)
                del input_ids, input_mask, label_ids, segment_ids
    torch.save(model_to_save.state_dict(), output_model_file)
    df = pd.read_csv('./data/data_KFold/data_origin_0/test.csv', encoding="utf-8", sep='\t', header=None)
    df.columns = ['query_id', 'reply_id', 'query', 'reply']
    res_proba = res_proba / 5
    df['logits_0'] = res_proba[:,0]
    df['logits_1'] = res_proba[:,1]
    df['label'] = np.argmax(res_proba, axis=1)
    df[['query_id', 'reply_id', 'label']].to_csv( os.path.join(output_dir, "submission.csv"), index=False, sep="\t", header=None)
    df[['query_id', 'reply_id', 'logits_0', 'logits_1']].to_csv( os.path.join(output_dir, "logits_test.csv"), index=False, sep="\t", header=None)


if __name__ == "__main__":
    main()
