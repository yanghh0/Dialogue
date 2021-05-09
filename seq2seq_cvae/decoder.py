

import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq2seq_cvae.config import Config


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        if Config.dec_num_layer > 1:
            self.dec_init_state_net = nn.ModuleList([
                nn.Linear(Config.dec_inputs_size, Config.dec_hidden_size) for i in range(Config.dec_num_layer)])
        else:
            self.dec_init_state_net = nn.Linear(Config.dec_inputs_size, Config.dec_hidden_size)

        self.decoder = self.get_rnncell("gru", Config.dec_input_embedding_size, Config.dec_hidden_size, Config.dec_num_layer, Config.keep_prob)
        self.fc = nn.Linear(Config.dec_hidden_size, Config.word_vocab_size)

    @staticmethod
    def get_rnncell(cell_type, input_size, hidden_size, num_layer, keep_prob, bidirectional=False):
        cell = getattr(nn, cell_type.upper())(
                    input_size=input_size, 
                    hidden_size=hidden_size, 
                    num_layers=num_layer, 
                    dropout=1-keep_prob, 
                    bidirectional=bidirectional, 
                    batch_first=True
                )
        return cell

    def forward(self, inputs, enc_outputs, inputs_length, context_vector):
        if Config.dec_num_layer > 1:
            dec_init_state = [self.dec_init_state_net[i](enc_outputs) for i in range(Config.dec_num_layer)]
            dec_init_state = torch.stack(dec_init_state)
        else:
            dec_init_state = self.dec_init_state_net(enc_outputs).unsqueeze(0)

        inputs = F.dropout(inputs, 1 - Config.keep_prob, self.training)
        if context_vector is not None:
            inputs = torch.cat([inputs, context_vector.unsqueeze(1).expand(inputs.size(0), inputs.size(1), context_vector.size(1))], 2)

        sorted_lens, len_ix = inputs_length.sort(0, descending=True)

        # Used for later reorder
        # 假如 x 原来的位置是3， 排序后变成 2，这个 2 就是排名，那么还原时位置 3 就需要用位置2的元素赋值
        inv_ix = len_ix.clone()
        inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

        # The number of inputs that have lengths > 0
        valid_num = torch.sign(sorted_lens).long().sum().item()
        zero_num = inputs.size(0) - valid_num

        sorted_inputs = inputs[len_ix].contiguous()
        sorted_init_state = dec_init_state[:, len_ix].contiguous()

        packed_inputs = pack_padded_sequence(sorted_inputs[:valid_num], list(sorted_lens[:valid_num]), batch_first=True)
        outputs, state = self.decoder(packed_inputs, sorted_init_state[:, :valid_num])

        # Reshape *final* output to (batch_size, hidden_size)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Add back the zero lengths
        if zero_num > 0:
            outputs = torch.cat([outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)
            state = torch.cat([state, sorted_init_state[:, valid_num:]], 1)

        # Reorder to the original order
        outputs = outputs[inv_ix].contiguous()
        state = state[:, inv_ix].contiguous()

        # compensate the last last layer dropout.
        state = F.dropout(state, self.decoder.dropout, self.training)
        outputs = F.dropout(outputs, self.decoder.dropout, self.training)

        outputs = self.fc(outputs)

        return outputs, state, None


if __name__ == "__main__":
    model = DecoderRNN()
    model.cuda()