
import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq2seq_gru.attention import LuongAttention
from seq2seq_gru.embedding import WordEmbedding


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(input_size=hidden_size, 
                          hidden_size=hidden_size, 
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, inputs, inputs_length, hidden=None):
        """shape of inputs: (t, b)
           shape of inputs_length: (b,)
        """
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, inputs_length)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, vocab_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=hidden_size, 
                          hidden_size=hidden_size, 
                          num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.attn = LuongAttention("dot", hidden_size)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, pre_hidden, encoder_outputs):
        """shape of input_step: (t=1, b)
           shape of encoder_outputs: (t, b, h)
        """
        embeded = self.embedding(input_step)
        embeded = self.embedding_dropout(embeded)

        # rnn_output: (1, b, h)
        rnn_output, hidden = self.gru(embeded, pre_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # shape of attn_weights: (b, 1, t)
        # shape of encoder_outputs:  (t, b, h)
        # shape of context: (b, 1, h) 
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)

        concat_output = torch.tanh(self.concat(concat_input))  
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden


def test_encoder():
    inputs = torch.LongTensor([[1, 2, 3, 4],
                               [4, 3, 2, 0],
                               [7, 4, 0, 0]])
    inputs_length = [4, 3, 2]
    use_gpu = True
    inputs = inputs.t()

    embedding = WordEmbedding(vocab_size=10, embedding_dim=6)
    encoder = EncoderRNN(hidden_size=6, embedding=embedding)
    if use_gpu:
        encoder.cuda()
        inputs = inputs.cuda()
    out = encoder(inputs, inputs_length)
    return out


def test_decoder(encoder_outputs):
    inputs = torch.LongTensor([[1, 2, 3, 4],
                               [4, 3, 2, 0],
                               [7, 4, 0, 0]])
    use_gpu = True
    inputs = inputs.t()

    embedding = WordEmbedding(vocab_size=10, embedding_dim=6)
    decoder = LuongAttnDecoderRNN(embedding=embedding, hidden_size=6, vocab_size=10)
    if use_gpu:
        decoder.cuda()
        inputs = inputs.cuda()
    out = decoder(inputs[0].unsqueeze(0), None, encoder_outputs)


if __name__ == "__main__":
    inputs = torch.LongTensor([[1, 2, 3, 4],
                               [4, 3, 2, 0],
                               [7, 4, 0, 0]])
    inputs_length = [4, 3, 2]
    use_gpu = True
    inputs = inputs.t()  # (b,t) -> (t,b)

    embedding = WordEmbedding(vocab_size=10, embedding_dim=6)
    encoder = EncoderRNN(hidden_size=6, embedding=embedding)
    decoder = LuongAttnDecoderRNN(embedding=embedding, hidden_size=6, vocab_size=10)
    if use_gpu:
        encoder.cuda()
        decoder.cuda()
        inputs = inputs.cuda()
    encoder_outputs = encoder(inputs, inputs_length)[0]
    out = decoder(inputs[0].unsqueeze(0), None, encoder_outputs)
