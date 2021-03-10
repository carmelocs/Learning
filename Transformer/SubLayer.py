import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Compute Scaled Dot Product Attention

        Input:
            query: [B, len_q, d_k]
            key: [B, len_k, d_k]
            value: [B, len_v, d_v]  # len_k == len_v
            mask: [B, len_q, len_k]
        Output:
            att: [B, len_q, d_v]
            scores: [B, len_q, len_k]
        """

        d_k = query.size(-1)  # [B, len_q, d_k]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            d_k)  # [B, len_q, len_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = scores.softmax(-1)

        if self.dropout is not None:
            scores = self.dropout(scores)

        att = torch.matmul(scores, value)  # [B, len_q, d_v]

        return att, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_head == 0

        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq = nn.Linear(d_model, num_head * d_k)
        self.wk = nn.Linear(d_model, num_head * d_k)
        self.wv = nn.Linear(d_model, num_head * d_v)
        self.fc = nn.Linear(num_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        '''
        Multi head attention with Residual connection and Layer Normalization

        Input:
            query: [B, len_q, d_k]
            key: [B, len_k, d_k]
            value: [B, len_v, d_v]  # len_k == len_v
            mask: [B, len_q, len_k]
        Output:
            output: [B, len_q, d_model]
            scores: [B, num_head, len_q, len_k]
        '''

        batch_size = query.size(0)

        residual = query  # [B, len_q, d_model]

        # print(f"query: {query.shape}, \nkey: {key.shape}, \nvalue: {value.shape}")

        # linear projection and split d_model by heads
        query = self.wq(query).view(batch_size,
                                    -1, self.num_head, self.d_k).transpose(
                                        1, 2)  # [B, num_head, len_q, d_k]
        key = self.wk(key).view(batch_size, -1, self.num_head,
                                self.d_k).transpose(
                                    1, 2)  # [B, num_head, len_k, d_k]
        value = self.wv(value).view(batch_size,
                                    -1, self.num_head, self.d_v).transpose(
                                        1, 2)  # [B, num_head, len_v, d_v]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)

        att, scores = self.attention(
            query, key, value, mask
        )  # att: [B, num_head, len_q, d_v], scores: [B, num_head, len_q, len_k]
        # print(f"scores: {scores.shape}, \nattention: {att.shape}")

        # concat heads
        att_cat = att.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_head * self.d_v)  # [B, len_q, d_model]

        # final linear projection
        output = self.fc(att_cat)  # [B, len_q, d_model]

        # dropout
        output = self.dropout(output)  # [B, len_q, d_model]

        # add residual and norm layer
        output = self.layer_norm(output + residual)  # [B, len_q, d_model]

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
        '''
        Position-wise Feed-Forward Networks with Residual connection and Layer Normalization
        Input:
            x: [B, len_seq, d_model]
        Output:
            x: [B, len_seq, d_model]
        '''

        residual = x  # [B, len_seq, d_model]

        x = self.w2(self.relu(self.w1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)  # [B, len_seq, d_model]

        return x


if __name__ == '__main__':
    # Make some fake data
    torch.manual_seed(0)

    BATCH_SIZE = 16
    MAX_LEN_SEQ = 100
    LEN_SRC = 100
    LEN_TGT = 120
    D_WORD_VEC = 512

    src_word = torch.rand(BATCH_SIZE, LEN_SRC).long()
    print(f"source word: {src_word.shape}")
    tgt_word = torch.rand(BATCH_SIZE, LEN_TGT).long()
    print(f"target word: {tgt_word.shape}")

    src_word_emb = nn.Embedding(LEN_SRC, D_WORD_VEC)
    tgt_word_emb = nn.Embedding(LEN_TGT, D_WORD_VEC)
    # query = src_word_emb(src_word)
    # key = src_word_emb(src_word)
    # value = src_word_emb(src_word)
    enc_input = src_word_emb(src_word)
    print(f"encoder input: {enc_input.shape}")
    dec_input = tgt_word_emb(tgt_word)
    print(f"decoder input: {dec_input.shape}")

    # Hyperparameters

    # number of encoder/decoder layers
    NUM_LAYER = 6

    # The dimensionality of input and output for EncoderDecoder model
    D_MODEL = 512

    # number of heads/parallel attention layers
    NUM_HEAD = 8

    # The dimensionality of qurey and key in each head
    D_K = D_MODEL // NUM_HEAD

    # The dimensionality of value in each head (could be different from d_k)
    D_V = D_K

    # The dimensionality of inner-layer for Position-wise Feed-Forward Network(FFN)
    D_FF = 2048

    multi_head = MultiHeadAttention(D_MODEL, NUM_HEAD, D_K, D_V)
    output, *_ = multi_head(dec_input, enc_input, enc_input)
    print(f"output of multi head attention: {output.shape}")

    ffn = PositionWiseFeedForward(D_MODEL, D_FF)
    ffn_output = ffn(output)
    print(f"output of Position-wise Feed-Forward: {ffn_output.shape}")
