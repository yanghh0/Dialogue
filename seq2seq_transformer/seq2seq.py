
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
from seq2seq_transformer.attention import MultiHeadAttention
from seq2seq_transformer.feed_forward import PositionwiseFeedForward
from seq2seq_transformer.pos_encoding import PositionalEncoding
from seq2seq_transformer.mask import get_pad_mask, get_subsequent_mask


class EncoderLayer(nn.Module):
    """Compose with two layers
    """

    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_inputs, slf_attn_mask=None):
        enc_outputs, enc_slf_attn = self.slf_attn(enc_inputs, enc_inputs, enc_inputs, mask=slf_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, enc_slf_attn


class DecoderLayer(nn.Module):
    """Compose with three layers
    """

    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_inputs, enc_outputs, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_outputs, dec_slf_attn = self.slf_attn(dec_inputs, dec_inputs, dec_inputs, mask=slf_attn_mask)
        dec_outputs, dec_enc_attn = self.enc_attn(dec_outputs, enc_outputs, enc_outputs, mask=dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    """A encoder model with self attention mechanism.
    """

    def __init__(self, vocab_size, d_model, d_inner, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seqs, src_mask, return_attns=False):
        enc_outputs = self.embedding(src_seqs)
        enc_outputs = self.position_enc(enc_outputs)
        enc_outputs = self.dropout(enc_outputs)
        enc_outputs = self.layer_norm(enc_outputs)

        for enc_layer in self.layer_stack:
            enc_outputs, enc_slf_attn = enc_layer(enc_outputs, slf_attn_mask=src_mask)

        return enc_outputs


class Decoder(nn.Module):
    """A decoder model with self attention mechanism.
    """

    def __init__(self, vocab_size, d_model, d_inner, n_head, n_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, trg_seqs, trg_mask, enc_outputs, src_mask, return_attns=False):
        dec_outputs = self.embedding(trg_seqs)
        dec_outputs = self.position_enc(dec_outputs)
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.layer_norm(dec_outputs)

        for dec_layer in self.layer_stack:
            dec_outputs, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_outputs, enc_outputs, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

        return self.fc(dec_outputs)


if __name__ == "__main__":
    vocab_size = 10
    d_model = 512
    d_inner = 2048
    n_head = 8
    n_layers = 6
    use_gpu = True

    src_seqs = torch.LongTensor([[1, 2, 3, 4],
                                 [4, 3, 2, 0],
                                 [7, 4, 0, 0]])
    trg_seqs = torch.LongTensor([[3, 1, 3, 4],
                                 [4, 6, 2, 0],
                                 [7, 7, 0, 0]])

    src_mask = get_pad_mask(src_seqs, pad_idx=0)
    trg_mask = get_pad_mask(trg_seqs, pad_idx=0) & get_subsequent_mask(trg_seqs)

    encoder = Encoder(vocab_size, d_model, d_inner, n_head, n_layers)
    decoder = Decoder(vocab_size, d_model, d_inner, n_head, n_layers)
    if use_gpu:
        encoder.cuda()
        decoder.cuda()
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        src_mask = src_mask.cuda()
        trg_mask = trg_mask.cuda()

    enc_outputs = encoder(src_seqs, src_mask)
    dec_outputs = decoder(trg_seqs, trg_mask, enc_outputs, src_mask)
