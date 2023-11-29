import torch
import pickle
import os
import json
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F

class MultiModalDataset(Dataset):
    def __init__(self, data_pth, vid_pth, aud_pth, txt_pth):
        self.conv_utt_id_list, self.conv_couples_list, self.y_emotions_list, \
        self.y_causes_list, self.conv_len_list, self.conv_id_list \
        = self.read_data(data_pth)
    
        self.vid_pth = vid_pth
        self.aud_pth = aud_pth
        self.txt_pth = txt_pth

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        conv_couples, y_emotions, y_causes = self.conv_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        conv_len, conv_id = self.conv_len_list[idx], self.conv_id_list[idx]
        conv_utt_ids = self.conv_utt_id_list[idx]

        assert conv_len == len(y_emotions)
        assert conv_len == len(conv_utt_ids)

        v, a, t, v_lens, a_lens, t_lens = self.load_tensors(conv_id, conv_utt_ids)

        return conv_couples, y_emotions, y_causes, conv_len, conv_id, \
               v, a, t, v_lens, a_lens, t_lens

    def load_tensors(self, conv_id, utt_ids):
      videos, v_lens = [], []
      audios, a_lens = [], []
      texts, t_lens = [], []

      for utt in utt_ids:
        id = 'dia'+str(conv_id)+'utt'+str(utt)+'.pkl'

        v = torch.tensor(pickle.load(open(os.path.join(self.vid_pth,id), "rb")), dtype=torch.float)
        a = pickle.load(open(os.path.join(self.aud_pth,id), "rb")).detach()
        t = pickle.load(open(os.path.join(self.txt_pth,id), "rb")).squeeze().detach()

        videos.append(v); v_lens.append(len(v))
        audios.append(a); a_lens.append(len(a))
        texts.append(t); t_lens.append(len(t))

      videos = pad_sequence(videos, batch_first=True) # [Nu, Nv, 1024]
      audios = pad_sequence(audios, batch_first=True)
      texts = pad_sequence(texts, batch_first=True)

      return videos, audios, texts, v_lens, a_lens, t_lens

    def get_mask(self, seq, lens):
      n, max_len, dim = seq.size()
      mask = np.zeros([n, max_len])
      for idx, seq_len in enumerate(lens):
        mask[idx][:seq_len] = 1

      return torch.BoolTensor(mask)


    def read_data(self, data_pth):
        data = json.load(open(data_pth, "r"))
        conv_id_list = []
        conv_len_list = []
        conv_utt_id_list = []
        conv_couples_list = []
        y_emotions_list, y_causes_list = [], []

        for conv in data:
          if len(conv["emotion-cause_pairs"]) != 0:
            conv_id_list.append(conv["conversation_ID"])
            utterances = conv["conversation"]
            conv_len = len(utterances)
            conv_len_list.append(conv_len)
            conv_utt_id_list.append([u['utterance_ID'] for u in utterances])

            couples = conv["emotion-cause_pairs"]

            conv_couples = [[int(e.split('_')[0]), int(c)] for e, c in couples]
            conv_emotions, conv_causes = zip(*conv_couples)
            conv_couples_list.append(conv_couples)

            y_emotions, y_causes = [], []
            for i in range(conv_len):
                emotion_label = int(i + 1 in conv_emotions)
                cause_label = int(i + 1 in conv_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)

        return conv_utt_id_list, conv_couples_list, y_emotions_list, y_causes_list, conv_len_list, conv_id_list,


def build_loaders(configs):
    dataset = MultiModalDataset(configs.anno_pth, configs.vid_pth, 
                                configs.aud_pth, configs.txt_pth)

    gen = torch.Generator().manual_seed(configs.seed)
    train_dataset, val_dataset = random_split(dataset, [1-configs.val_ratio, configs.val_ratio], gen)

    train_loader = DataLoader(dataset=train_dataset, 
                             batch_size=configs.batch_size,
                             shuffle=configs.shuffle, 
                             collate_fn=batch_preprocessing,
                             num_workers=configs.num_worker)

    val_loader = DataLoader(dataset=val_dataset, 
                           batch_size=configs.batch_size,
                           shuffle=False, 
                           collate_fn=batch_preprocessing,
                           num_workers=configs.num_worker)
    
    return train_loader, val_loader

# collate function
def batch_preprocessing(batch):
    conv_couples_b, y_emotions_b, y_causes_b, conv_len_b, conv_id_b, \
    v_token_b, a_token_b, t_token_b, v_lens_b, a_lens_b, t_lens_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b = pad_convs(conv_len_b, y_emotions_b, y_causes_b)
    adj_b = pad_matrices(conv_len_b)
    v_token_b = pad_conversations(list(v_token_b), conv_len_b)
    a_token_b = pad_conversations(list(a_token_b), conv_len_b)
    t_token_b = pad_conversations(list(t_token_b), conv_len_b)

    v_mask = get_mask(v_token_b, v_lens_b) # [Nc, Nu, Nv]
    a_mask = get_mask(a_token_b, a_lens_b)
    t_mask = get_mask(t_token_b, t_lens_b)

    return np.array(conv_len_b), np.array(adj_b), \
           np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), conv_couples_b, conv_id_b, \
           v_token_b, a_token_b, t_token_b, v_mask, a_mask, t_mask


def pad_conversations(seq_tokens, conv_lens):
  num_conv = len(seq_tokens) # N
  num_utt = max(conv_lens) # U
  num_tokens = max([s.size()[1] for s in seq_tokens]) # [Nc, Nu, Nv, dim]

  for i, seq in enumerate(seq_tokens):
    cur_utt, cur_len, _ = seq.size()
    pad = (0, 0, 0, num_tokens - cur_len, 0, num_utt - cur_utt)
    seq_tokens[i] = F.pad(seq, pad, "constant", 0)

  return pad_sequence(seq_tokens, batch_first=True)


def get_mask(seq, lens):
    num_conv, num_utt, max_len, dim = seq.size()
    mask = np.zeros([num_conv, num_utt, max_len])

    for conv, conv_len in enumerate(lens):
      for utt, seq_len in enumerate(conv_len):
        mask[conv][utt][:seq_len] = 1

    return torch.BoolTensor(mask)


def pad_convs(conv_len_b, y_emotions_b, y_causes_b):
    max_conv_len = max(conv_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_conv_len, -1)
        y_causes_ = pad_list(y_causes, max_conv_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_


def pad_matrices(conv_len_b):
    N = max(conv_len_b)
    adj_b = []
    for conv_len in conv_len_b:
        adj = np.ones((conv_len, conv_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad
