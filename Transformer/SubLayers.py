import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    
    """
    Compute Scaled Dot Product Attention
    """

    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
    
        d_k = query.size(-1) # [batch, len, d_k]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = scores.softmax(-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        att = torch.matmul(scores, value)

        return att, scores

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0

        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.wq = nn.Linear(d_model, h * d_k)
        self.wk = nn.Linear(d_model, h * d_k)
        self.wv = nn.Linear(d_model, h * d_v)
        self.fc = nn.Linear(h * d_v, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)
        
        residual = query # [batch, len_seq, d_model]

        # print(f"query: {query.shape}, \nkey: {key.shape}, \nvalue: {value.shape}")

        # linear projection and split d_model by heads
        query = self.wq(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # [batch, h, len_query, d_k]
        key = self.wk(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) # [batch, h, len_key, d_k]
        value = self.wv(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2) # [batch, h, len_value, d_v]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)

        att, scores = self.attention(query, key, value, mask) # att: [batch, h, len_seq, d_v], scores: [batch, h, len_seq, len_seq]
        # print(f"scores: {scores.shape}, \nattention: {att.shape}")

        # concat heads
        att_cat = att.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v) # [batch, len_seq, d_model]

        # final linear projection
        output = self.fc(att_cat) # [batch, len_seq, d_model]

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(output + residual)

        return output, scores

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        # self.w1 = nn.Conv1d(d_model, d_ff, 1)
        # self.w2 = nn.Conv1d(d_model, d_ff, 1)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w2(self.relu(self.w1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x

