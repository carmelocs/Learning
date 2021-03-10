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
        '''
        Encoder Layer: Encoder self-attention and 
                        Position-wise Feed-Forward layers
        
        Input:
            enc_input: [B, len_src, d_model]
            slf_att_mask: [B, len_src, len_src]
        Output:
            enc_output: [B, len_src, d_model]
            enc_slf_att: [B, len_src, len_src]
        '''

        # print(f"enc_layer_slf_mask: {slf_att_mask.shape}")
        enc_output, enc_slf_att = self.slf_att(enc_input,
                                               enc_input,
                                               enc_input,
                                               mask=slf_att_mask)
        enc_output = self.pos_ffn(enc_output)
        # print(f"enc_layer_output: {enc_output.shape}")
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
                dec_slf_mask=None,
                dec_enc_mask=None):
        '''
        Decoder Layer: Decoder self-attention, Decoder-Encoder attention and 
                        Position-wise Feed-Forward layers
        
        Input:
            dec_input: [B, len_tgt, d_model]
            enc_output: [B, len_src, d_model]
            dec_slf_mask: [B, len_tgt, len_tgt]
            dec_enc_mask: [B, len_tgt, len_src]
        Output:
            dec_output: [B, len_tgt, d_model]
        '''
        
        dec_output, dec_slf_att = self.slf_att(dec_input,
                                               dec_input,
                                               dec_input,
                                               dec_slf_mask)
        # [B, len_tgt, d_model]
        
        dec_output, dec_enc_att = self.enc_att(dec_output,
                                               enc_output,
                                               enc_output,
                                               dec_enc_mask)
        # [B, len_tgt, d_model]

        dec_output = self.pos_ffn(dec_output)
        # print(f"decoder output: {dec_output.shape}")
        
        return dec_output, dec_slf_att, dec_enc_att

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

    encoder = EncoderLayer(D_MODEL, NUM_HEAD, D_K, D_V, D_FF)
    enc_output, *_ = encoder(enc_input)

    decoder = DecoderLayer(D_MODEL, NUM_HEAD, D_K, D_V, D_FF)
    dec_output, *_ = decoder(dec_input, enc_output)
