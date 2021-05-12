

import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq2seq_cvae.config import Config


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        if Config.dec_num_layer > 1:
            self.dec_init_state_net = nn.ModuleList([
                nn.Linear(Config.dec_inputs_size, Config.dec_hidden_size) for i in range(Config.dec_num_layer)])
        else:
            self.dec_init_state_net = nn.Linear(Config.dec_inputs_size, Config.dec_hidden_size)

        self.decoder = self.get_rnncell(
            "gru", 
            Config.dec_input_embed_size, 
            Config.dec_hidden_size, 
            Config.dec_num_layer, 
            Config.keep_prob
        )
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

    def train_loop(self, inputs, dec_init_state, inputs_length, attr_vector):
        inputs = F.dropout(inputs, 1 - Config.keep_prob, self.training)
        if attr_vector is not None:
            inputs = torch.cat([inputs, attr_vector.unsqueeze(1).expand(inputs.size(0), inputs.size(1), attr_vector.size(1))], 2)

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

    def inference_loop(self, dec_init_state, attr_vector, word_embedding):
        batch_size = dec_init_state.size(1)
        cell_state, cell_input, cell_output = dec_init_state, None, None

        outputs = []
        context_state = []

        for time in range(Config.max_utt_length + 1):
            if cell_output is None:
                next_input_id = dec_init_state.new_full((batch_size,), SOS_token, dtype=torch.long)
                done = dec_init_state.new_zeros(batch_size, dtype=torch.bool)
                cell_state = dec_init_state
            else:
                cell_output = self.fc(cell_output)
                outputs.append(cell_output)

                next_input_id = torch.max(cell_output, 1)[1]
                next_input_id = next_input_id * (~done).long() # make sure the next_input_id to be 0 if done
                done = (next_input_id == EOS_token) | done
                context_state.append(next_input_id)

            next_input = word_embedding(next_input_id)
            if attr_vector is not None:
                next_input = torch.cat([next_input, attr_vector], 1)
            if done.long().sum() == batch_size:
                break

            cell_output, cell_state = self.decoder(next_input.unsqueeze(1), cell_state)
            # Squeeze the time dimension
            cell_output = cell_output.squeeze(1)

            # zero out done sequences
            cell_output = cell_output * (~done).float().unsqueeze(1)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), \
               cell_state, \
               torch.cat([_.unsqueeze(1) for _ in context_state], 1)

    def forward(self, mode, inputs, enc_outputs, inputs_length, attr_vector, word_embedding):
        """shape of inputs: (60, 40, 200)
           shape of attr_vector: (60, 30)
        """
        if Config.dec_num_layer > 1:
            dec_init_state = [self.dec_init_state_net[i](enc_outputs) for i in range(Config.dec_num_layer)]
            dec_init_state = torch.stack(dec_init_state)
        else:
            dec_init_state = self.dec_init_state_net(enc_outputs).unsqueeze(0)

        if mode == "test":
            return self.inference_loop(dec_init_state, attr_vector, word_embedding)
        else:
            return self.train_loop(inputs, dec_init_state, inputs_length, attr_vector)


if __name__ == "__main__":
    model = DecoderRNN()
    model.cuda()