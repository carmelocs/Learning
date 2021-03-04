import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import EncoderLayer, DecoderLayer


def get_pad_mask(seq_q, seq_k):
    # seq_k和seq_q的形状都是[batch, len_seq]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    # print(pad_mask.shape)
    pad_mask = pad_mask.unsqueeze(1).expand(
        -1, len_q, -1)  # shape [batch, len_seq, len_seq]
    return pad_mask


def get_subsequent_mask(len_q, len_k):
    "Mask out subsequent positions."
    return torch.triu(torch.ones(1, len_q, len_k), diagonal=1) == 0


class Encoder(nn.Module):
    def __init__(self,
                 len_src_vocab,
                 d_word_vec,
                 num_layer,
                 d_model,
                 h,
                 d_k,
                 d_v,
                 d_ff,
                 dropout=0.1):

        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(len_src_vocab, d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_ff, h, d_k, d_v, dropout)
            for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src_word, src_mask=None):

        enc_output = self.layer_norm(self.dropout(self.src_word_emb(src_word)))
        # print(f"enc_output: {enc_output.shape}")

        for enc_layer in self.layer_stack:
            enc_output, *_ = enc_layer(enc_output, src_mask)
        print(
            f"enc_output: {enc_output.shape}\nenc_src_mask: {src_mask.shape}")

        return enc_output


class Decoder(nn.Module):
    def __init__(self,
                 len_tgt_vocab,
                 d_word_vec,
                 num_layer,
                 d_model,
                 h,
                 d_k,
                 d_v,
                 d_ff,
                 dropout=0.1):

        super(Decoder, self).__init__()
        self.tgt_word_emb = nn.Embedding(len_tgt_vocab, d_word_vec)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_ff, h, d_k, d_v, dropout)
            for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tgt_word, enc_output, tgt_mask=None, src_mask=None):
        dec_output = self.layer_norm(self.dropout(self.tgt_word_emb(tgt_word)))
        # print(f"tgt_word: {tgt_word.shape}\nenc_output: {enc_output.shape}\ntgt_mask: {tgt_mask.shape}\nsrc_mask: {src_mask.shape}")

        for dec_layer in self.layer_stack:
            dec_output, *_ = dec_layer(dec_output,
                                       enc_output,
                                       slf_att_mask=tgt_mask,
                                       dec_enc_att_mask=src_mask)

        return dec_output


class Transformer(nn.Module):
    def __init__(self,
                 len_src_vocab,
                 len_tgt_vocab,
                 d_word_vec,
                 d_model,
                 d_ff,
                 num_layer,
                 h,
                 d_k,
                 d_v,
                 dropout=0.1):

        super(Transformer, self).__init__()

        self.len_src_vocab = len_src_vocab
        self.len_tgt_vocab = len_tgt_vocab
        self.d_model = d_model

        self.encoder = Encoder(len_src_vocab=len_src_vocab,
                               d_word_vec=d_word_vec,
                               num_layer=num_layer,
                               d_model=d_model,
                               h=h,
                               d_k=d_k,
                               d_v=d_v,
                               d_ff=d_ff,
                               dropout=dropout)

        self.decoder = Decoder(len_tgt_vocab=len_tgt_vocab,
                               d_word_vec=d_word_vec,
                               num_layer=num_layer,
                               d_model=d_model,
                               h=h,
                               d_k=d_k,
                               d_v=d_v,
                               d_ff=d_ff,
                               dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, len_tgt_vocab)

        assert d_word_vec == d_model

    def forward(self, src_word, tgt_word):

        src_mask = get_pad_mask(src_word, src_word)
        tgt_mask = get_pad_mask(tgt_word, tgt_word) & get_subsequent_mask(
            self.len_tgt_vocab, self.len_tgt_vocab)
        # print(f"trans_src_mask: {src_mask.shape}\ntrans_tgt_mask: {tgt_mask.shape}")

        enc_output = self.encoder(src_word=src_word, src_mask=src_mask)
        # print(f"trans_src_word: {src_word.shape}\ntrans_src_mask: {src_mask.shape}\ntransf_enc_output: {enc_output.shape}\n")
        dec_output = self.decoder(tgt_word=tgt_word,
                                  enc_output=enc_output,
                                  tgt_mask=tgt_mask,
                                  src_mask=src_mask)
        print(
            f"trans_src_word: {src_word.shape}\ntrans_src_mask: {src_mask.shape}\ntransf_enc_output: {enc_output.shape}\ntrans_dec_output: {dec_output.shape}"
        )
        seq_pred = self.tgt_word_prj(dec_output)
        print(f"seq_pred: {seq_pred.shape}")

        return F.log_softmax(seq_pred, dim=-1).max(-1)[0]
