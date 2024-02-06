import numpy as np
import torch

# seq_len = 5
# base_idx = np.arange(1, seq_len + 1)
# print(base_idx)
# emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
# print(emo_pos)
# cau_pos = np.concatenate([base_idx] * seq_len, axis=0)
# print(cau_pos)

# emo_pos = torch.LongTensor(emo_pos)
# cau_pos = torch.LongTensor(cau_pos)


# if seq_len > 4:
#     emo_pos_mask = np.array(list(map(lambda x: 1 <= x <= 4, emo_pos.tolist())), dtype=int)
#     cau_pos_mask = np.array(list(map(lambda x: 1 <= x <= 4, cau_pos.tolist())), dtype=int)
#     cau_pos_mask = torch.BoolTensor(cau_pos_mask)
#     emo_pos_mask = torch.BoolTensor(emo_pos_mask)
#     emo_pos = emo_pos.masked_select(emo_pos_mask)
#     cau_pos = cau_pos.masked_select(cau_pos_mask)
# print(emo_pos)
# print(cau_pos)

# in couple generator
# in_emb = [[[1,2,3], 
#            [9,4,0]],
#           [[5,7,8], 
#            [6,9,0]]]
# in_emb = torch.as_tensor(in_emb)
# bs, seq_len, in_dim = in_emb.size()
# p_left = torch.cat([in_emb] * seq_len, dim=2)
# p_left = p_left.reshape(-1, seq_len * seq_len, in_dim)
# p_right = torch.cat([in_emb] * seq_len, dim=1)
# p = torch.cat([p_left, p_right], dim=2)
# p = p.view(bs, seq_len, seq_len, 2 * in_dim)
# print(p)

# tensor = torch.randn((4, 10, 5))  # Replace this with your actual tensor

# Calculate the length of each sequence along the second dimension (s)
# seq_lengths = torch.sum(tensor.sum(dim=-1) != 0, dim=-1)

# print(seq_lengths)

t = torch.tensor([[[1,1,0],
                   [1,0,0]],
                  [[1,1,1],
                   [1,1,0]]
                  ])

