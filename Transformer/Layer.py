import torch
import torch.nn as nn
from SubLayer import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_k, d_v, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_att = MultiHeadAttention(d_model,
                                          num_head,
                                          d_k,
                                          d_v,
                                          dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, enc_input, slf_att_mask=None):
        enc_output, enc_slf_att = self.slf_att(enc_input,
                                               enc_input,
                                               enc_input,
                                               mask=slf_att_mask)
        enc_output = self.pos_ffn(enc_output)
        # print(f"enc_layer_slf_mask: {slf_att_mask.shape}\nenc_layer_output: {enc_output.shape}")
        return enc_output, enc_slf_att


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_k, d_v, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_att = MultiHeadAttention(d_model,
                                          num_head,
                                          d_k,
                                          d_v,
                                          dropout=dropout)
        self.enc_att = MultiHeadAttention(d_model,
                                          num_head,
                                          d_k,
                                          d_v,
                                          dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self,
                dec_input,
                enc_output,
                slf_att_mask=None,
                dec_enc_att_mask=None):
        dec_output, dec_slf_att = self.slf_att(dec_input,
                                               dec_input,
                                               dec_input,
                                               mask=slf_att_mask)
        dec_output, dec_enc_att = self.enc_att(dec_output,
                                               enc_output,
                                               enc_output,
                                               mask=dec_enc_att_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_att, dec_enc_att
