import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import BertModel
import numpy as np
import geoopt
import itertools

class GAT(nn.Module):
    """References: https://github.com/Determined22/Rank-Emotion-Cause/blob/master/src/networks/rank_cp.py"""
    def __init__(self, num_layers, num_heads_per_layer, num_features_per_layer, feat_dim, device,  dropout=0.6, bias=True):
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
        out_emb = F.dropout(out_emb, self.attn_dropout) # (b, N, i)

        return out_emb

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

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        # self.activation = nn.ReLU()
        # self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        # x = self.activation(x)
        # x = self.out(x)
        return x

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, device):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
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

class EmotionRecognitionModel(nn.Module):
    def __init__(self, args):
        super(EmotionRecognitionModel, self).__init__()

        self.args = args
        num_features_per_layer_gat = [args.num_features_per_layer_gat] * args.num_layers_gat
        num_heads_per_layer_gat = [args.num_heads_per_layer_gat] * args.num_layers_gat
        num_bilstm_layers = 2
        num_classes = 7

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.gnn = GAT(args.num_layers_gat, num_heads_per_layer_gat, num_features_per_layer_gat, args.input_dim_transformer, args.device)
        self.emotion_classifier = EmotionClassifier(args.input_dim_transformer, args.input_dim_transformer // 2, 7)
        # self.emotion_classifier = BiLSTMClassifier(args.input_dim_transformer, args.input_dim_transformer // 2, num_bilstm_layers, num_classes, self.args.device)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len, adj, y_mask_b):
        bert_output = self.bert_model(input_ids=bert_token_b.to(self.args.device),
                                attention_mask=bert_masks_b.to(self.args.device),
                                token_type_ids=bert_segment_b.to(self.args.device)
                                )
        convo_utt_embeddings = self.batched_index_select(bert_output, bert_utt_b.to(self.args.device))
        modified_embeddings_gat = self.gnn(convo_utt_embeddings, convo_len, adj)
        y_mask_b_new = y_mask_b.unsqueeze(2).expand(-1, -1, self.args.input_dim_transformer)
        modified_embeddings_gat = modified_embeddings_gat.masked_select(y_mask_b_new).reshape(-1, 768)
        emotion_pred = self.emotion_classifier(modified_embeddings_gat)

        return emotion_pred

    def batched_index_select(self, bert_output, bert_utt_b):
        # bert_output = (bs, convo_len, hidden_dim), bert_utt_b = (bs, convo_len)=idx of cls tokens
        hidden_state = bert_output[0]
        dummy = bert_utt_b.unsqueeze(2).expand(bert_utt_b.size(0), bert_utt_b.size(1), hidden_state.size(2)) # same as .expand(bs, convo_len, hidden_dim)
        # Use gather to select specific elements from the hidden state tensor based on indices provided by bert_utt_b
        convo_utt_embeddings = hidden_state.gather(1, dummy) # convo shape = dummy shape = (bs, convo_len, hidden_dim)
        return convo_utt_embeddings

    # Define new function for aggregating last 4 hidden layer output for getting emb

