

import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq2seq_cvae.config import Config


class UtteranceRNN(nn.Module):
    def __init__(self):
        super(UtteranceRNN, self).__init__()
        self.utt_encoder = self.get_rnncell("gru", Config.utt_input_size, Config.utt_hidden_size, Config.utt_num_layer, Config.keep_prob, True)

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

    def forward(self, inputs, length_mask=None):
        if length_mask is None:
            length_mask = torch.sum(torch.sign(torch.max(torch.abs(inputs), 2)[0]), 1)
            length_mask = length_mask.long()

        sorted_lens, len_ix = length_mask.sort(0, descending=True)

        # Used for later reorder
        # 假如 x 原来的位置是3， 排序后变成 2，这个 2 就是排名，那么还原时位置 3 就需要用位置2的元素赋值
        inv_ix = len_ix.clone()
        inv_ix[len_ix] = torch.arange(0, len(len_ix)).type_as(inv_ix)

        # The number of inputs that have lengths > 0
        valid_num = torch.sign(sorted_lens).long().sum().item()
        zero_num = inputs.size(0) - valid_num

        sorted_inputs = inputs[len_ix].contiguous()
        packed_inputs = pack_padded_sequence(sorted_inputs[:valid_num], list(sorted_lens[:valid_num]), batch_first=True)

        outputs, state = self.utt_encoder(packed_inputs)

        # Reshape *final* output to (batch_size, hidden_size)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Add back the zero lengths
        if zero_num > 0:
            outputs = torch.cat([outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)
            state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

        # Reorder to the original order
        outputs = outputs[inv_ix].contiguous()
        state = state[:, inv_ix].contiguous()

        # compensate the last last layer dropout.
        outputs = F.dropout(outputs, self.utt_encoder.dropout, self.training)
        state = F.dropout(state, self.utt_encoder.dropout, self.training)

        # get only the last layer
        encoded_input = torch.cat([state[-2], state[-1]], 1)
        
        return encoded_input, self.utt_encoder.hidden_size * 2


if __name__ == "__main__":
    model = UtteranceRNN()
    model.cuda()