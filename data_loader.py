import os
from os.path import join
import sys
import json
import numpy as np
import scipy.sparse as sp
import torch
from  torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import *
from transformers import BertTokenizer

def build_train_data(args, fold_id, shuffle=True):
    train_dataset = CustomDataset(args, fold_id, data_type='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader

def build_inference_data(args, fold_id, data_type, shuffle=True):
    dataset = CustomDataset(args, fold_id, data_type=data_type)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader

class CustomDataset(Dataset):
    def __init__(self, args, fold_id, data_type):
        self.input_dir = args.input_dir
        self.text_input_dir = args.text_input_dir
        self.data_type = data_type
        self.split = args.kfold
        self.emotion_idx = dict(zip(['anger', 'disgust', 'fear', 'sadness', 'neutral','joy','surprise'], range(7)))

        self.train_file = join(self.input_dir, self.text_input_dir, f"split{self.split}", "fold{}_train.json".format(fold_id))
        self.val_file = join(self.input_dir, self.text_input_dir, f"split{self.split}", "fold{}_val.json".format(fold_id))
        if data_type == 'test':
            self.test_file = join(self.input_dir, self.text_input_dir, f"split{self.split}", "test.json")

        self.batch_size = args.batch_size
        self.epochs = args.num_epochs
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.given_emotion_idxs, self.y_emotions_labels, self.y_causes_labels, \
        self.convo_len_list, \
        self.bert_token_list, self.bert_utt_idx_list, self.bert_segments_idx_list, \
        self.bert_convo_num_tokens_list = self.read_json_file(self.data_type)

    def __len__(self):
        return len(self.y_causes_labels)

    def __getitem__(self, idx):
        given_emotion_idx = self.given_emotion_idxs[idx]
        y_emotions, y_causes = self.y_emotions_labels[idx], self.y_causes_labels[idx]
        convo_len =self.convo_len_list[idx]
        bert_token, bert_utt_idx = self.bert_token_list[idx], self.bert_utt_idx_list[idx]
        bert_segments_idx, bert_convo_num_tokens = self.bert_segments_idx_list[idx], self.bert_convo_num_tokens_list[idx]

        if bert_convo_num_tokens > 512:
            given_emotion_idx, \
            y_emotions, y_causes, \
            convo_len, \
            bert_token, bert_utt_idx, \
            bert_segments_idx, bert_convo_num_tokens = self.token_trunk(given_emotion_idx, y_emotions, y_causes, convo_len, bert_token, bert_utt_idx, bert_segments_idx, bert_convo_num_tokens)

        bert_token = torch.LongTensor(bert_token)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_utt_idx = torch.LongTensor(bert_utt_idx)

        assert convo_len == len(y_emotions)
        return given_emotion_idx, y_emotions, y_causes, convo_len, \
                bert_token, bert_segments_idx, bert_utt_idx, bert_convo_num_tokens

    def read_json_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.val_file
        elif data_type == 'test':
            data_file = self.test_file

        with open(data_file, 'r') as f:
            data_list = json.load(f)
        given_emotion_idxs = []
        convo_len_list = []
        y_emotions_labels, y_causes_labels = [], []
        bert_token_list = []
        bert_utt_idx_list = []
        bert_segments_idx_list = []
        bert_convo_num_tokens_list = []

        for convo in data_list:
            # Store the given target emotion utterance index in conversation, 0 based
            emo_utt_num = int(convo['emotion_utterance_ID'].split('utt')[1])
            given_emotion_idxs.append(emo_utt_num - 1)

            convo_len = len(convo['conversation'])
            convo_len_list.append(convo_len)

            # For every sample given store whether each utt is a cause: true(1) or false(0) 
            y_causes = np.zeros(convo_len)
            for c in convo['cause_utterances']:
                y_causes[int(c) - 1] = 1.

            y_emotions = []

            doc_str = ''
            for utt in convo['conversation']:
                y_emotions.append(self.emotion_idx[utt['emotion']])

                doc_str += '[CLS] ' + utt['text'] + ' [SEP] '

            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)

            utt_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            convo_num_tokens = len(indexed_tokens)

            segment_ids = []
            segment_indices = [i for i, x in enumerate(indexed_tokens) if x == 101] # CLS=101 SEP=102
            segment_indices.append(len(indexed_tokens))
            # Keeping segment ids alternating 0s and 1s, but since convo between multiple ppl might be a good idea to keep all 0s?
            for i in range(len(segment_indices) - 1):
                segment_len = segment_indices[i + 1] - segment_indices[i]
                if i % 2 == 0:
                    segment_ids.extend([0] * segment_len)
                else:
                    segment_ids.extend([1] * segment_len)

            assert len(utt_indices) == convo_len, f'utt indices len != convo len'
            assert len(segment_ids) == len(indexed_tokens), f'len utt ids != len indexed tokens'
            bert_token_list.append(indexed_tokens)
            bert_utt_idx_list.append(utt_indices)
            bert_segments_idx_list.append(segment_ids)
            bert_convo_num_tokens_list.append(convo_num_tokens)

            y_emotions_labels.append(y_emotions)
            y_causes_labels.append(y_causes)

        return given_emotion_idxs, y_emotions_labels, y_causes_labels, convo_len_list, \
                bert_token_list, bert_utt_idx_list, bert_segments_idx_list, bert_convo_num_tokens_list

    def token_trunk(self, given_emotion_idx, y_emotions, y_causes, convo_len, bert_tokens, bert_utt_idx, bert_segments_idx, bert_convo_num_tokens):
        # Try to keep the half which contains the given emotion

        if given_emotion_idx >= convo_len / 2: # this means emotion is in the second half, we should keep the second half of tokens, until the end
            i = 0
            while True:
                temp_bert_token = bert_tokens[bert_utt_idx[i]: ] # bert_utt_idx are idxs of cls only, we're trying to choose the first cls from where num of tokens <= 512
                if len(temp_bert_token) <= 512:
                    cls_idx = bert_utt_idx[i]
                    bert_tokens = bert_tokens[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx: ]
                    bert_utt_idx = [p - cls_idx for p in bert_utt_idx[i: ]]
                    given_emotion_idx = given_emotion_idx - i
                    y_emotions = y_emotions[i: ]
                    y_causes = y_causes[i: ]
                    convo_len = convo_len - i
                    break
                i = i + 1

        if given_emotion_idx < convo_len / 2:
            i = convo_len - 1 # num of cls = convo len, i gives last cls, i => utt
            while True:
                temp_bert_token = bert_tokens[ :bert_utt_idx[i]] #bert_utt_idx[i] not included
                if len(temp_bert_token) <= 512:
                    cls_idx = bert_utt_idx[i] # => tokens in utt
                    bert_tokens = bert_tokens[ :cls_idx]
                    bert_segments_idx = bert_segments_idx[ :cls_idx]
                    bert_utt_idx = bert_utt_idx[ :i] # don't include ith one
                    y_emotions = y_emotions[ :i]
                    y_causes = y_causes[ :i]
                    convo_len = i
                    break
                i = i - 1
        return given_emotion_idx, y_emotions, y_causes, convo_len, bert_tokens, bert_utt_idx, bert_segments_idx, bert_convo_num_tokens

def bert_batch_preprocessing(batch):
    given_emotion_idxs_b, y_emotions_b, y_causes_b, convo_len_b, \
    bert_token_b, bert_segment_b, bert_utt_b, bert_convo_num_tokens_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b = pad_convo(convo_len_b, y_emotions_b, y_causes_b)
    adj_b = pad_matrices(convo_len_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_utt_b = pad_sequence(bert_utt_b, batch_first=True, padding_value=0)

    batch_size, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([batch_size, max_len], dtype=float)

    for convo_id, convo_num_tokens in enumerate(bert_convo_num_tokens_b):
        bert_masks_b[convo_id][ :convo_num_tokens] = 1

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape

    return np.array(given_emotion_idxs_b), np.array(adj_b), np.array(convo_len_b), \
        np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), \
        bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b

def pad_convo(convo_len_b, y_emotions_b, y_causes_b):
    max_convo_len = max(convo_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_convo_len, -1)
        y_causes_ = pad_list(y_causes, max_convo_len, 0)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_

def pad_matrices(convo_len_b):
    N = max(convo_len_b)
    adj_b = []
    for convo_len in convo_len_b:
        adj = np.ones((convo_len, convo_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad = np.concatenate((element_list_pad, pad_mark_list))
    return element_list_pad.tolist()

