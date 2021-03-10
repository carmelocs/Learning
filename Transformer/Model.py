import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import EncoderLayer, DecoderLayer


def get_pad_mask(seq_q, seq_k):
    '''
    Input:
        seq_q: [B, len_q]
        seq_k: [B, len_k]
    Output:
        pad_mask: [B, len_q, len_k]
    '''

    len_q = seq_q.size(1)

    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    # print(f"pad mask: {pad_mask.shape}")

    pad_mask = pad_mask.unsqueeze(1).repeat(
        1, len_q, 1)  # shape [B, len_q, len_k]
    # print(f"output of pad mask:{pad_mask.shape}")

    return pad_mask


def get_subsequent_mask(len_q, len_k):
    '''
    Mask out subsequent positions.
    Input:
        int: len_q, len_k
    Output:
        subsequent_mask: [1, len_q, len_k]
    '''
    return torch.triu(torch.ones(1, len_q, len_k), diagonal=1) == 0


class Encoder(nn.Module):
    def __init__(self,
                 len_src_vocab,
                 d_word_vec,
                 d_model,
                 num_head,
                 num_layer,
                 d_k,
                 d_v,
                 d_ff,
                 dropout=0.1):

        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(len_src_vocab, d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, num_head, d_k, d_v, d_ff, dropout)
            for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src_word, src_slf_mask=None):
        '''
        Input:
            src_word: [B, len_src]
            src_slf_mask: [B, len_src, len_src]
        Output:
            enc_output: [B, len_src, d_model]
        '''

        enc_output = self.layer_norm(self.dropout(self.src_word_emb(src_word)))
        # print(f"enc_output: {enc_output.shape}")

        for enc_layer in self.layer_stack:
            enc_output, *_ = enc_layer(enc_output, src_slf_mask)
        # print(
            # f"enc_output: {enc_output.shape}\nenc_src_mask: {src_mask.shape}")

        return enc_output


class Decoder(nn.Module):
    def __init__(self,
                 len_tgt_vocab,
                 d_word_vec,
                 d_model,
                 num_head,
                 num_layer,
                 d_k,
                 d_v,
                 d_ff,
                 dropout=0.1):

        super(Decoder, self).__init__()
        self.tgt_word_emb = nn.Embedding(len_tgt_vocab, d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, num_head, d_k, d_v, d_ff, dropout)
            for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tgt_word, enc_output, tgt_slf_mask=None, tgt_src_mask=None):
        '''
        Input:
            tgt_word: [B, len_tgt]
            enc_output: [B, len_src, d_model]
            tgt_slf_mask: [B, len_tgt, len_tgt]
            tgt_src_mask: [B, len_tgt, len_src]
        Output:
            dec_output: [B, len_tgt, d_model]
        '''

        dec_output = self.layer_norm(self.dropout(self.tgt_word_emb(tgt_word)))
        # print(f"tgt_word: {tgt_word.shape}\nenc_output: {enc_output.shape}\ntgt_mask: {tgt_mask.shape}\nsrc_mask: {src_mask.shape}")

        for dec_layer in self.layer_stack:
            dec_output, *_ = dec_layer(dec_output,
                                       enc_output,
                                       dec_slf_mask=tgt_slf_mask,
                                       dec_enc_mask=tgt_src_mask)

        return dec_output


class Transformer(nn.Module):
    def __init__(self,
                 len_src_vocab,
                 len_tgt_vocab,
                 d_word_vec,
                 d_model,
                 num_head,
                 num_layer,
                 d_k,
                 d_v,
                 d_ff,
                 dropout=0.1):

        super(Transformer, self).__init__()

        self.len_src_vocab = len_src_vocab
        self.len_tgt_vocab = len_tgt_vocab
        self.d_model = d_model

        self.encoder = Encoder(len_src_vocab=len_src_vocab,
                               d_word_vec=d_word_vec,
                               d_model=d_model,
                               num_head=num_head,
                               num_layer=num_layer,
                               d_k=d_k,
                               d_v=d_v,
                               d_ff=d_ff,
                               dropout=dropout)

        self.decoder = Decoder(len_tgt_vocab=len_tgt_vocab,
                               d_word_vec=d_word_vec,
                               d_model=d_model,
                               num_head=num_head,
                               num_layer=num_layer,
                               d_k=d_k,
                               d_v=d_v,
                               d_ff=d_ff,
                               dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, len_tgt_vocab)

        assert d_word_vec == d_model

    def forward(self, src_word, tgt_word):
        '''
        Input:
            src_word: [B, len_src]
            tgt_word: [B, len_tgt]
        Output:
            pred: [B, len_tgt]
        '''

        src_slf_mask = get_pad_mask(src_word, src_word)
        tgt_slf_mask = get_pad_mask(tgt_word, tgt_word) & get_subsequent_mask(
            self.len_tgt_vocab, self.len_tgt_vocab)
        tgt_src_mask = get_pad_mask(tgt_word, src_word)
        print(f"src_slf_mask: {src_slf_mask.shape}\ntgt_slf_mask: {tgt_slf_mask.shape}\ntgt_src_mask: {tgt_src_mask.shape}")

        enc_output = self.encoder(src_word=src_word, src_slf_mask=src_slf_mask)
        # print(f"trans_src_word: {src_word.shape}\ntrans_src_mask: {src_mask.shape}\ntransf_enc_output: {enc_output.shape}\n")
        dec_output = self.decoder(tgt_word=tgt_word,
                                  enc_output=enc_output,
                                  tgt_slf_mask=tgt_slf_mask,
                                  tgt_src_mask=tgt_src_mask)
        print(
            f"trans_src_word: {src_word.shape}\nsrc_slf_mask: {src_slf_mask.shape}\ntransf_enc_output: {enc_output.shape}\ntrans_dec_output: {dec_output.shape}"
        )
        seq_pred = self.tgt_word_prj(dec_output)
        print(f"seq_pred: {seq_pred.shape}")

        return F.log_softmax(seq_pred, dim=-1).max(-1)[0]


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

    # enc_pad_mask = get_pad_mask(src_word, src_word)

    # dec_enc_pad_mask = get_pad_mask(tgt_word, src_word)

    # dec_pad_mask = get_pad_mask(tgt_word, tgt_word)
    # dec_sub_mask = get_subsequent_mask(LEN_TGT, LEN_TGT)
    # dec_slf_att_mask = dec_pad_mask & dec_sub_mask

    transformer = Transformer(
        len_src_vocab=LEN_SRC,
        len_tgt_vocab=LEN_TGT,
        d_word_vec=D_WORD_VEC,
        d_model=D_MODEL,
        num_head=NUM_HEAD,
        num_layer=NUM_LAYER,
        d_k=D_K,
        d_v=D_V,
        d_ff=D_FF,
    )
    pred = transformer(src_word, tgt_word)
    print(f"pred: {pred.shape}")
