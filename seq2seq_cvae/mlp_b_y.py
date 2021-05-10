

import sys
sys.path.append(".")
sys.path.append("..")


import torch
import torch.nn as nn
from seq2seq_cvae.config import Config


class MLPby(nn.Module):
    def __init__(self):
        super(MLPby, self).__init__()
        self.bow_project = nn.Sequential(
            nn.Linear(Config.gen_inputs_size, 400),
            nn.Tanh(),
            nn.Dropout(1 - Config.keep_prob),
            nn.Linear(400, Config.word_vocab_size)
        )
        if Config.use_hcf:
            self.act_project = nn.Sequential(
                nn.Linear(Config.gen_inputs_size, 400),
                nn.Tanh(),
                nn.Dropout(1 - Config.keep_prob),
                nn.Linear(400, Config.act_vocab_size)
            )

    def forward(self, inputs):
        """shape of inputs: (60, 30 + 4 + 4 + 600 + 200)
        """
        bow_logits = self.bow_project(inputs)
        if Config.use_hcf:
            act_logits = self.act_project(inputs)
        else:
            act_logits = inputs.new_zeros(Config.batch_size, Config.act_vocab_size)
        return bow_logits, act_logits


if __name__ == "__main__":
    model = MLPby()
    model.cuda()