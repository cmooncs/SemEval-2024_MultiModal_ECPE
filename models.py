import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import BertModel
import numpy as np

class EmbeddingModifierTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(EmbeddingModifierTransformer, self).__init__()

        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)

        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        self.num_layers = num_layers

    def forward(self, input_embeddings):
        # Apply multiple layers of self-attention and feedforward networks
        modified_embeddings = input_embeddings  # initialize
        for _ in range(self.num_layers):
            # Multi-Head Self-Attention, giving k, q, v
            attn_output, _ = self.self_attn(modified_embeddings, modified_embeddings, modified_embeddings)
            # Residual connection
            modified_embeddings = self.layer_norm1(attn_output + modified_embeddings)

            # Position-wise Feedforward Network
            ff_output = self.feedforward(modified_embeddings)
            # Residual connection
            modified_embeddings = self.layer_norm2(ff_output + modified_embeddings)

        return modified_embeddings

class MultipleCauseClassifier(nn.Module):
    """Gives probability for each pair within each conversation whether that utt-pair has a cause"""
    def __init__(self, input_dim, num_utt_tensors, num_labels):
        super(MultipleCauseClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_utt_tensors = num_utt_tensors

        # Linear layers for each tensor
        self.linear_layers = nn.ModuleList([nn.Linear(self.input_dim, 1) for _ in range(self.num_utt_tensors)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        probabilities = []
        for tensor, linear in zip(x, self.linear_layers):
            tensor_probs = linear(tensor)
            # The output will have shape (35, 1) change to (35)
            tensor_probs = tensor_probs.squeeze()
            tensor_probs = self.sigmoid(tensor_probs)
            probabilities.append(tensor_probs)
        probabilities = torch.stack(probabilities)
        return probabilities

class SingleCauseClassifier(nn.Module):
    """Gives probability for given pair whether that utt-pair has a cause"""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(SingleCauseClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # probabilities = torch.sigmoid(x)
        # return probabilities
        return x

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
        self.dropout = int(dropout)

        for i in range(self.num_layers):
            in_dim = self.gnn_dims[i] * self.num_heads_per_layer[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layers.append(GraphAttentionLayer(self.num_heads_per_layer[i], in_dim, self.gnn_dims[i + 1], self.dropout, self.device))

    def forward(self, embeddings, convo_len, adj):
        batches, max_convo_len, _ = embeddings.size()
        assert max(convo_len) == max_convo_len

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
        h = torch.matmul(in_emb_, self.W)

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
        attn.data.masked_fill_(mask.bool(), -np.inf)

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

class EmotionCausePairClassifierModel(nn.Module):
    def __init__(self, args):
        super(EmotionCausePairClassifierModel, self).__init__()

        self.args = args
        args.classifier_hidden_dim1 = 768
        args.classifier_hidden_dim2 = 384

        num_features_per_layer_gat = [args.num_features_per_layer_gat] * args.num_layers_gat
        num_heads_per_layer_gat = [args.num_heads_per_layer_gat] * args.num_layers_gat

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.transformer_model = EmbeddingModifierTransformer(args.input_dim_transformer, args.hidden_dim_transformer, args.num_heads_transformer, args.num_layers_transformer)
        self.gnn = GAT(args.num_layers_gat, num_heads_per_layer_gat, num_features_per_layer_gat, args.input_dim_transformer, args.device)
        self.classifier = SingleCauseClassifier(args.input_dim_transformer * 2, args.classifier_hidden_dim1, args.classifier_hidden_dim2)

    def forward(self, emotion_idxs, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len, adj):
        bert_output = self.bert_model(input_ids=bert_token_b.to(self.args.device),
                                attention_mask=bert_masks_b.to(self.args.device),
                                token_type_ids=bert_segment_b.to(self.args.device)
                                )
        convo_utt_embeddings = self.batched_index_select(bert_output, bert_utt_b.to(self.args.device))
        # modified_embeddings = self.transformer_model(convo_utt_embeddings)
        modified_embeddings_gat = self.gnn(convo_utt_embeddings, convo_len, adj)
        # Create pair of given emotion utt with all other utt in a convo
        utt_pairs = []
        for idx, convo in enumerate(modified_embeddings_gat):
            emotion_id = emotion_idxs[idx]
            emotion_utt_emb = convo[emotion_id]
            repeated_emotion_utt_emb = emotion_utt_emb.repeat(len(convo), 1) # along dim 1
            pairs = torch.cat((convo, repeated_emotion_utt_emb), dim=1)
            utt_pairs.append(pairs)
        # Convert to torch tensor
        utt_pairs = torch.stack(utt_pairs)
        # .view(bs * n, input_dim) : 
        bs = utt_pairs.shape[0]
        n = utt_pairs.shape[1]
        utt_pairs = utt_pairs.view(bs * n, utt_pairs.shape[2])
        # Classify each pair as having cause or not 
        probabilities = self.classifier(utt_pairs)
        # .view(bs, n)
        return probabilities.view(bs, n)

    def batched_index_select(self, bert_output, bert_utt_b):
        # bert_output = (bs, convo_len, hidden_dim), bert_utt_b = (bs, convo_len)=idx of cls tokens
        hidden_state = bert_output[0]
        dummy = bert_utt_b.unsqueeze(2).expand(bert_utt_b.size(0), bert_utt_b.size(1), hidden_state.size(2)) # same as .expand(bs, convo_len, hidden_dim)
        # Use gather to select specific elements from the hidden state tensor based on indices provided by bert_utt_b
        convo_utt_embeddings = hidden_state.gather(1, dummy) # convo shape = dummy shape = (bs, convo_len, hidden_dim)
        return convo_utt_embeddings

    # Define new function for aggregating last 4 hidden layer output for getting emb

