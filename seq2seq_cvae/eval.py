
import sys
sys.path.append(".")
sys.path.append("..")


import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from utils.Sw_data import Data, DataLoader
from seq2seq_cvae.kg_cvae import KgRnnCVAE
from seq2seq_cvae.config import Config
from utils.functions import normalizeStringNLTK, get_bleu_stats, get_checkpoint_state, pad_to


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class Chatbot:
    def __init__(self, corpus_file):
        self.data_obj = Data(corpus_file)
        self.data_loader = DataLoader(self.data_obj, "test")
        self.model = KgRnnCVAE()

        ckpt = get_checkpoint_state("checkpoint")
        if ckpt:
            print("Reading dm models parameters from %s" % ckpt)
            self.model.load_state_dict(torch.load(ckpt))

        if Config.use_gpu:
            self.model.cuda()

        self.data_loader.epoch_init()
        ref_batch_data = self.data_loader.next_batch()

        self.repeat = 5
        # shape=(None, None, Config.max_utt_length)
        self.inputs_contexts = torch.zeros((1, 8, Config.max_utt_length), dtype=torch.long)
        # shape=(None,)
        self.context_lens = torch.tensor([8], dtype=torch.long)
        # shape=(None, None)
        self.floors = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.long)
        # shape=(None,)
        self.topics = torch.tensor([np.random.randint(0, self.data_obj.topic_alphabet.num_tokens - 1)], dtype=torch.long)
        # shape=(None, 4)
        self.my_profile = ref_batch_data[4]
        # shape=(None, 4)
        self.ot_profile = ref_batch_data[5]
        # shape=(None, None)
        self.output_tokens = torch.tensor([[SOS_token, EOS_token]], dtype=torch.long)
        # shape=(None,)
        self.output_lens = torch.tensor([2], dtype=torch.long)
        # shape=(None,)
        self.output_des = torch.tensor([1], dtype=torch.long)

    def eval(self, sentence, dest=sys.stdout):
        sentence = normalizeStringNLTK(sentence).split(" ")
        sentence_to_indexes = self.data_obj.word_alphabet.list_to_indexes(sentence)
        sentence_to_indexes = pad_to(sentence_to_indexes, Config.max_utt_length)

        temp_inputs_contexts = self.inputs_contexts.clone()
        self.inputs_contexts[0, 0:self.context_lens[0] - 1, :] = temp_inputs_contexts[0, 1:self.context_lens[0], :]
        self.inputs_contexts[0, self.context_lens[0] - 1, :] = torch.LongTensor(sentence_to_indexes)

        batch_data = self.inputs_contexts, \
                     self.context_lens,    \
                     self.floors, \
                     self.topics, \
                     self.my_profile, \
                     self.ot_profile, \
                     self.output_tokens, \
                     self.output_lens, \
                     self.output_des

        feed_dict = self.model.batch_feed(batch_data, use_prior=True, repeat=self.repeat)
        with torch.no_grad():
            _, _, _, _, act_logits, _, _, _, _, dec_out_words = self.model(feed_dict, mode='test')

        word_outs = dec_out_words.cpu().numpy()
        act_logits = act_logits.cpu().numpy()
        sample_words = np.split(word_outs, self.repeat, axis=0)
        sample_des = np.split(act_logits, self.repeat, axis=0)


        word_vocab = self.data_obj.word_alphabet.index2token
        topic_vocab = self.data_obj.topic_alphabet.index2token
        act_vocab = self.data_obj.act_alphabet.index2token

        local_tokens = []
        local_tokens_indexes = []
        for r_id in range(self.repeat):
            pred_outs = sample_words[r_id]
            pred_act = np.argmax(sample_des[r_id], axis=1)[0]
            pred_tokens = [word_vocab[e] for e in pred_outs[0].tolist() if e != EOS_token and e != 0]
            pred_str = " ".join(pred_tokens).replace(" ' ", "'")
            dest.write("Sample %d (%s) >> %s\n" % (r_id, act_vocab[pred_act], pred_str))
            local_tokens.append(pred_str)
            local_tokens_indexes.append(pred_outs[0].tolist())

        dest.write("\n")

        choice = np.random.randint(0, self.repeat)
        answer_tokens = local_tokens[choice]
        answer_tokens_indexes = local_tokens_indexes[choice]

        print(answer_tokens)
        print(answer_tokens_indexes)

        answer_tokens_indexes = pad_to(answer_tokens_indexes, Config.max_utt_length)
        temp_inputs_contexts = self.inputs_contexts.clone()
        self.inputs_contexts[0, 0:self.context_lens[0] - 1, :] = temp_inputs_contexts[0, 1:self.context_lens[0], :]
        self.inputs_contexts[0, self.context_lens[0] - 1, :] = torch.LongTensor(answer_tokens_indexes)

        return answer_tokens


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")

    cbt = Chatbot(corpus_file)
    
    input_sentence = ''
    while True:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit':
            break
        
        print("user 0: %s" % input_sentence)
        answer = cbt.eval(input_sentence)
        print("chat bot: %s" % answer)