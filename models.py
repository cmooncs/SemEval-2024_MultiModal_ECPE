import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import BertModel
import numpy as np
import geoopt
import itertools

# class EmbeddingModifierTransformer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
#         super(EmbeddingModifierTransformer, self).__init__()

#         self.self_attn = nn.MultiheadAttention(input_dim, num_heads)

#         self.feedforward = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )

#         self.layer_norm1 = nn.LayerNorm(input_dim)
#         self.layer_norm2 = nn.LayerNorm(input_dim)

#         self.num_layers = num_layers

#     def forward(self, input_embeddings):
#         # Apply multiple layers of self-attention and feedforward networks
#         modified_embeddings = input_embeddings  # initialize
#         for _ in range(self.num_layers):
#             # Multi-Head Self-Attention, giving k, q, v
#             attn_output, _ = self.self_attn(modified_embeddings, modified_embeddings, modified_embeddings)
#             # Residual connection
#             modified_embeddings = self.layer_norm1(attn_output + modified_embeddings)

#             # Position-wise Feedforward Network
#             ff_output = self.feedforward(modified_embeddings)
#             # Residual connection
#             modified_embeddings = self.layer_norm2(ff_output + modified_embeddings)

#         return modified_embeddings

# class MultipleCauseClassifier(nn.Module):
#     """Gives probability for each pair within each conversation whether that utt-pair has a cause"""
#     def __init__(self, input_dim, num_utt_tensors, num_labels):
#         super(MultipleCauseClassifier, self).__init__()
#         self.input_dim = input_dim
#         self.num_utt_tensors = num_utt_tensors

#         self.linear_layers = nn.ModuleList([nn.Linear(self.input_dim, 1) for _ in range(self.num_utt_tensors)])
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         probabilities = []
#         for tensor, linear in zip(x, self.linear_layers):
#             tensor_probs = linear(tensor)
#             # The output will have shape (35, 1) change to (35)
#             tensor_probs = tensor_probs.squeeze()
#             tensor_probs = self.sigmoid(tensor_probs)
#             probabilities.append(tensor_probs)
#         probabilities = torch.stack(probabilities)
#         return probabilities

class GAT(nn.Module):
    """References: https://github.com/Determined22/Rank-Emotion-Cause/blob/master/src/networks/rank_cp.py"""
    def __init__(self, num_layers, num_heads_per_layer, num_features_per_layer, feat_dim, device,  dropout=0.1, bias=True):
        super(GAT, self).__init__()
        assert num_layers == len(num_heads_per_layer) == len(num_features_per_layer), f'Enter valid architecture parameters for GAT'
        in_dim = feat_dim
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.gnn_dims = [in_dim] + [int(i) for i in num_features_per_layer]
        self.bias = bias
        self.gnn_layers = nn.ModuleList()
        self.device = device
        self.dropout = dropout

        for i in range(self.num_layers):
            in_dim = self.gnn_dims[i] * self.num_heads_per_layer[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layers.append(GraphAttentionLayer(self.num_heads_per_layer[i], in_dim, self.gnn_dims[i + 1], self.dropout, self.device))

    def forward(self, embeddings, convo_len, adj):
        batches, max_convo_len, _ = embeddings.size()
        assert np.max(convo_len) == max_convo_len

        for i, gnn_layer in enumerate(self.gnn_layers):
            embeddings = gnn_layer(embeddings, adj)

        return embeddings

class GraphAttentionLayer(nn.Module):
    def __init__(self, num_heads, f_in, f_out, attn_dropout, device, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = f_in
        self.out_dim = f_out
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.device = device

        self.W = nn.Parameter(torch.Tensor(self.num_heads, self.in_dim, self.out_dim))
        self.b = nn.Parameter(torch.Tensor(self.out_dim))
        self.a_src = nn.Parameter(torch.Tensor(self.num_heads, self.out_dim, 1)) #for Whi
        self.a_dst = nn.Parameter(torch.Tensor(self.num_heads, self.out_dim, 1)) #for Whj

        self.init_gnn_param()

        assert self.in_dim == self.num_heads * self.out_dim
        self.H = nn.Linear(self.in_dim, self.in_dim)
        init.xavier_normal_(self.H.weight)

    def init_gnn_param(self):
        init.xavier_uniform_(self.W.data)
        init.zeros_(self.b.data)
        init.xavier_uniform_(self.a_src.data)
        init.xavier_uniform_(self.a_dst.data)

    def forward(self, in_emb, adj=None):
        #batch, N, in_dim = in_emb.size() # N = max convo len, num of utt, NxN matrix
        batch, N, in_dim = in_emb.size() # N = max convo len, num of utt, NxN matrix

        assert in_dim == self.in_dim # (b, N, i)

        # Batched matrix multiplication, non-matrix(batch) dimensions are broadcasted
        in_emb_ = in_emb.unsqueeze(1) # (b, 1, N, in_dim)
        # (b, 1, N, in) x (attn_heads, in, out) = (b, attn_heads, N, out)
        # project the input features to 'attn_heads' independent output features
        h = torch.matmul(in_emb_, self.W)

        # instead of taking dot product bw [x,y] and a, take dp bw x and a_src and y and a_trg
        attn_src = torch.matmul(F.tanh(h), self.a_src) # (b, attn_heads, N, 1)
        attn_dst = torch.matmul(F.tanh(h), self.a_dst) # (b, attn_heads, N, 1)
        # repeat value in the last dimension N times in the 4th dimension
        # add each N value to every other N value
        attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)

        # This does the same, but a broadcasted addition
        # attn = attn_src + attn_dst.permute(0, 1, 3, 2) # (batch, attn_heads, N, N)
        attn = F.leaky_relu(attn, 0.2, inplace=True)

        adj = torch.FloatTensor(adj).to(self.device)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask.bool(), -1e20)

        attn = F.softmax(attn, dim=-1) # dim=-1 means apply along last dim, gives attn coeff
        out_emb = torch.matmul(attn, h) + self.b # (b, attn_heads, N, out)
        # transpose => (b, N, attn_heads, out)
        # but transpose still refers to the same memory whose order is diff, hence call contiguous()
        # since view only operates on contiguous memory
        out_emb = out_emb.transpose(0, 1).contiguous().view(batch, N, -1) # (b, N, attn_heads*out)
        out_emb = F.elu(out_emb) # exponential linear unit, have negative values, allows to push mean closer to 0, allows faster convergence

        # Skip connection with learnable weights
        gate = F.sigmoid(self.H(in_emb))
        out_emb = gate * out_emb + (1 - gate) * in_emb # broadcast (b, 1, N, i), (b, N, i)
        out_emb = F.dropout(out_emb, self.attn_dropout, training=self.training) # (b, N, i)

        return out_emb

class CauseEmotionClassifier(nn.Module):
    """Gives probability for given pair whether that utt-pair has a cause"""
    def __init__(self, input_dim, hidden_dim):
        super(CauseEmotionClassifier, self).__init__()

        self.emotion_fc1 = nn.Linear(input_dim, hidden_dim)
        self.emotion_fc2 = nn.Linear(hidden_dim, 1)
        self.cause_fc1 = nn.Linear(input_dim, hidden_dim)
        self.cause_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x1 = self.emotion_fc1(x)
        x1 = self.emotion_fc2(x1)
        x2 = self.cause_fc1(x)
        x2 = self.cause_fc2(x2)
        return x1.squeeze(2), x2.squeeze(2)

class HyperbolicLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, manifold):
        super(HyperbolicLinearLayer, self).__init__()

        self.W = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        nn.init.kaiming_uniform_(self.W.data)
        nn.init.zeros_(self.bias.data)
        self.manifold = manifold

    def forward(self, x):
        hyperbolic_output = self.manifold.mobius_matvec(self.W, x) + self.bias.unsqueeze(0)
        return hyperbolic_output

class HyperbolicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HyperbolicClassifier, self).__init__()

        self.ball = geoopt.PoincareBall()

        self.fc1 = HyperbolicLinearLayer(input_dim, hidden_dim, self.ball)
        self.fc2 = HyperbolicLinearLayer(hidden_dim, output_dim, self.ball)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, pos_emb_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, pos_emb_dim)

    def forward(self, positions):
        """
        Input: (batch_size, seq_lens) = positions in the sequence
        Output: (batch_size, seq_len, pos_emb_dim) = position embeddings
        """
        embeddings = self.pos_embedding(positions)
        return embeddings

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.device = device

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        out, state = self.lstm(x, (h0, c0))
        # print("out shape {}".format(out.shape))
        out_last_tstep = out[:, -1, :]
        logits = self.fc(out_last_tstep)

        return logits

class PairsClassifier(nn.Module):
    def __init__(self, args):
        super(PairsClassifier, self).__init__()
        self.max_seq_len = args.max_convo_len
        self.pos_emb_dim = args.pos_emb_dim
        self.input_emb_dim = args.input_dim_transformer
        self.concat_emb_dim = 2 * (self.input_emb_dim + self.pos_emb_dim)
        self.device = args.device

        self.pos_emb_layer = nn.Embedding(self.max_seq_len, self.pos_emb_dim)
        nn.init.xavier_uniform_(self.pos_emb_layer.weight)
        self.bilstm = BiLSTMClassifier(self.concat_emb_dim, self.concat_emb_dim // 2, 2, self.device)
        # self.hyperbolic_classifier = HyperbolicClassifier(self.concat_emb_dim, self.concat_emb_dim // 2, 1)

    def forward(self, input_embeddings, convo_len):
        pairs = self.couple_generator(input_embeddings, convo_len)

        pairs_pred = self.bilstm(pairs)
        # print("pairs pred shape {}".format(pairs_pred.shape))
        return pairs_pred.view(self.bs, self.seq_len * self.seq_len)

        # couples = self.euclidean_to_hyperbolic(couples)
        # couples_pred = self.hyperbolic_classifier(couples)
        # return couples_pred.squeeze(2), emo_cau_pos

    def euclidean_to_hyperbolic(self, embeddings, scale=0.01):
        ball = geoopt.PoincareBall()
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        hyperbolic_embeddings = ball.expmap0(scale * embeddings)

        return hyperbolic_embeddings

    def couple_generator(self, in_emb, convo_len):
        bs, seq_len, in_dim = in_emb.size()
        # create positional embeddings (need to add a check for max_seq_len)
        seq_lengths = torch.norm(in_emb, dim=-1).int().to(self.device) # shape (bs, seq_lens)
        # print("seq lengths dim {}".format(seq_lengths.shape))
        pos_embeddings = self.pos_emb_layer(seq_lengths)
        # print("pos emb dim {}".format(pos_embeddings.shape))
        in_emb = torch.concat([in_emb, pos_embeddings], dim=2)
        # print("in_emb size {}".format(in_emb.shape))
        self.bs, self.seq_len, self.in_dim = in_emb.size()

        p_left = torch.cat([in_emb] * self.seq_len, dim=2)
        p_left = p_left.reshape(-1, self.seq_len * self.seq_len, self.in_dim)
        p_right = torch.cat([in_emb] * self.seq_len, dim=1)
        p = torch.cat([p_left, p_right], dim=2)
        p = p.view(self.bs, self.seq_len, self.seq_len, 2 * self.in_dim)

        return p.view(self.bs * self.seq_len * self.seq_len, 2 * self.in_dim)

        # pairs_b = []
        # pairs_pos_b = []
        # batch_idxs_b = []
        # for batch_idx in range(len(indices_pred_e)):
        #     row1 = indices_pred_e[batch_idx]
        #     row2 = indices_pred_c[batch_idx]
        #     pos = list(itertools.product(row1, row2))
        #     # pos = [(x, y) for x in row1 for y in row2]
        #     if self.device == 'cpu':
        #         prs = [p[batch_idx][index].detach().numpy() for index in pos]
        #     else:
        #         prs = [p[batch_idx][index].detach().cpu().numpy() for index in pos]
        #     batch_idxs = [batch_idx] * len(prs)
        #     pairs_pos_b.extend(pos)
        #     pairs_b.extend(prs)
            # batch_idxs_b.extend(batch_idxs)

        # pairs_b = torch.as_tensor(pairs_b).to(self.device)
        # pairs_pos_b = torch.as_tensor(pairs_pos_b).to(self.device)
        # batch_b = torch.as_tensor(batch_idxs_b).to(self.device)
        # return pairs_b, pairs_pos_b, batch_idxs_b

        # if seq_len > self.max_seq_len:
        #     rel_mask = np.array(list(map(lambda x: -self.max_seq_len <= x <= self.max_seq_len, rel_pos.tolist())), dtype=int)
        #     rel_mask = torch.BoolTensor(rel_mask).to(self.device)
        #     rel_pos = rel_pos.masked_select(rel_mask)
        #     emo_pos = emo_pos.masked_select(rel_mask)
        #     cau_pos = cau_pos.masked_select(rel_mask)

        #     rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * in_dim)
        #     rel_mask = rel_mask.unsqueeze(0).expand(bs, -1, -1)
        #     p = p.masked_select(rel_mask)
        #     p = p.reshape(bs, -1, 2 * in_dim)

        # assert rel_pos.size(0) == p.size(1)
        # rel_pos = rel_pos.unsqueeze(0).expand(bs, -1)

        # emo_cau_pos = []
        # for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
        #     emo_cau_pos.append([emo, cau])
        # return p, rel_pos, emo_cau_pos

class EmotionCausePairExtractorModel(nn.Module):
    def __init__(self, args):
        super(EmotionCausePairExtractorModel, self).__init__()

        self.args = args
        self.threshold_emo = args.threshold_emo
        self.threshold_cau = args.threshold_cau
        args.classifier_hidden_dim1 = 384
        args.classifier_hidden_dim2 = 384

        num_features_per_layer_gat = [args.num_features_per_layer_gat] * args.num_layers_gat
        num_heads_per_layer_gat = [args.num_heads_per_layer_gat] * args.num_layers_gat

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # self.transformer_model = EmbeddingModifierTransformer(args.input_dim_transformer, args.hidden_dim_transformer, args.num_heads_transformer, args.num_layers_transformer)
        self.gnn = GAT(args.num_layers_gat, num_heads_per_layer_gat, num_features_per_layer_gat, args.input_dim_transformer, args.device)
        self.cause_emotion_classifier = CauseEmotionClassifier(args.input_dim_transformer, args.classifier_hidden_dim1)
        self.pairs_classifier = PairsClassifier(args)
        self.pos_emb_layer = PositionalEmbedding(args.max_convo_len, args.pos_emb_dim)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len, adj, y_mask_b):
        bert_output = self.bert_model(input_ids=bert_token_b.to(self.args.device),
                                attention_mask=bert_masks_b.to(self.args.device),
                                token_type_ids=bert_segment_b.to(self.args.device)
                                )
        convo_utt_embeddings = self.batched_index_select(bert_output, bert_utt_b.to(self.args.device))
        # modified_embeddings = self.transformer_model(convo_utt_embeddings)
        modified_embeddings_gat = self.gnn(convo_utt_embeddings, convo_len, adj)
        emotion_pred, cause_pred = self.cause_emotion_classifier(modified_embeddings_gat)

        # y_mask_b = torch.tensor(y_mask_b).bool().to(self.args.device)
        # convert the output to binary
        # binary_preds_e = (torch.sigmoid(emotion_pred) > self.threshold_emo).float()
        # indices_pred_e = []
        # print(binary_preds_e)
        # binary_preds_c = (torch.sigmoid(cause_pred) > self.threshold_cau).float()
        # indices_pred_c = []
        # print(binary_preds_c)
        # for idx, preds in enumerate(binary_preds_e):
        #     preds = preds.masked_select(y_mask_b[idx])
        #     indices = torch.nonzero(preds == 1.)
        #     indices_pred_e.append(indices)
        # for idx, preds in enumerate(binary_preds_c):
        #     preds = preds.masked_select(y_mask_b[idx])
        #     indices = torch.nonzero(preds == 1.)
        #     indices_pred_c.append(indices)

        pair_pred = self.pairs_classifier(modified_embeddings_gat, convo_len)
        return emotion_pred, cause_pred, pair_pred

    def batched_index_select(self, bert_output, bert_utt_b):
        # bert_output = (bs, convo_len, hidden_dim), bert_utt_b = (bs, convo_len)=idx of cls tokens
        hidden_state = bert_output[0]
        dummy = bert_utt_b.unsqueeze(2).expand(bert_utt_b.size(0), bert_utt_b.size(1), hidden_state.size(2)) # same as .expand(bs, convo_len, hidden_dim)
        # Use gather to select specific elements from the hidden state tensor based on indices provided by bert_utt_b
        convo_utt_embeddings = hidden_state.gather(1, dummy) # convo shape = dummy shape = (bs, convo_len, hidden_dim)
        return convo_utt_embeddings

    # Define new function for aggregating last 4 hidden layer output for getting emb

