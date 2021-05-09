
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
from seq2seq_cvae.config import Config
from seq2seq_cvae.context_encoder import ContextRNN
from seq2seq_cvae.utterance_encoder import UtteranceRNN
from seq2seq_cvae.decoder import DecoderRNN
from utils.Sw_data import Data
from utils.functions import sample_gaussian


class RecognitionNetwork(nn.Module):
    def __init__(self):
        super(RecognitionNetwork, self).__init__()
        self.recogNet = nn.Linear(Config.recog_input_size, Config.latent_size * 2)

    def forward(self, inputs):
        recog_mulogvar = self.recogNet(inputs)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2, 1)
        return recog_mu, recog_logvar


class PriorNetwork(nn.Module):
    def __init__(self):
        super(PriorNetwork, self).__init__()
        self.priorNet = nn.Sequential(
            nn.Linear(Config.prior_input_size, np.maximum(Config.latent_size * 2, 100)),
            nn.Tanh(),
            nn.Linear(np.maximum(Config.latent_size * 2, 100), Config.latent_size * 2)
        )

    def forward(self, inputs):
        prior_mulogvar = self.priorNet(inputs)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
        return prior_mu, prior_logvar


class KgRnnCVAE(nn.Module):
    def __init__(self):
        super(KgRnnCVAE, self).__init__()

        # all dialog context and known attributes
        self.input_contexts = tf.placeholder(dtype=tf.int64, shape=(None, None, Config.max_utt_length), name="dialog_context")
        self.floors = tf.placeholder(dtype=tf.int64, shape=(None, None), name="floor")
        self.context_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="context_lens")
        self.topics = tf.placeholder(dtype=tf.int64, shape=(None,), name="topics")
        self.my_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="my_profile")
        self.ot_profile = tf.placeholder(dtype=tf.float32, shape=(None, 4), name="ot_profile")

        # target response given the dialog context
        self.output_tokens = tf.placeholder(dtype=tf.int64, shape=(None, None), name="output_token")
        self.output_lens = tf.placeholder(dtype=tf.int32, shape=(None,), name="output_lens")
        self.output_des = tf.placeholder(dtype=tf.int64, shape=(None,), name="output_dialog_acts")

        self.topic_embedding = nn.Embedding(Config.topic_vocab_size, Config.topic_embed_size)
        if Config.use_hcf:
            self.act_embedding = nn.Embedding(Config.act_vocab_size, Config.act_embed_size)
        self.word_embedding = nn.Embedding(Config.word_vocab_size, Config.word_embed_size, padding_idx=0)

        self.utt_encoder = UtteranceRNN()
        self.ctx_encoder = ContextRNN()
        self.attribute_fc1 = nn.Sequential(
            nn.Linear(Config.act_embed_size, 30), 
            nn.Tanh()
        )
        self.recogNet_mulogvar = RecognitionNetwork()
        self.priorNet_mulogvar = PriorNetwork()
        self.bow_project = nn.Sequential(
            nn.Linear(Config.gen_inputs_size, 400),
            nn.Tanh(),
            nn.Dropout(1 - Config.keep_prob),
            nn.Linear(400, Config.word_vocab_size)
        )
        if Config.use_hcf:
            self.y_project = nn.Sequential(
                nn.Linear(Config.gen_inputs_size, 400),
                nn.Tanh(),
                nn.Dropout(1 - Config.keep_prob),
                nn.Linear(400, Config.act_vocab_size)
            )
        self.decoder = DecoderRNN()

    def batch_feed(self, batch_data, use_prior, repeat=1):
        context, context_lens, floors, topics, my_profiles, ot_profiles, outputs, output_lens, output_des = batch_data
        feed_dict = {
            "input_contexts": context, 
            "context_lens":context_lens,
            "floors": floors, 
            "topics":topics, 
            "my_profile": my_profiles,
            "ot_profile": ot_profiles, 
            "output_tokens": outputs,
            "output_lens": output_lens,
            "output_des": output_des,
            "use_prior": use_prior
        }
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key == "use_prior":
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1] * len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if Config.use_gpu:
            feed_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in feed_dict.items()}
        else:
            feed_dict = {k: v if isinstance(v, torch.Tensor) else v for k, v in feed_dict.items()}

        return feed_dict













        # self.bi_utt_enc_cell = self.get_rnncell("gru", Config.word_embed_size, Config.utt_hidden_size, 1, Config.keep_prob, True)
        # input_embedding_size = Config.utt_hidden_size * 2
        # output_embedding_size = Config.utt_hidden_size * 2

        # # ------------------contextRNN------------------
        # joint_embedding_size = input_embedding_size + 2
        # self.ctx_enc_cell = self.get_rnncell("gru", joint_embedding_size, Config.ctx_hidden_size, 1, Config.keep_prob)
        # self.attribute_fc1 = nn.Sequential(
        #     nn.Linear(Config.act_embed_size, 30), 
        #     nn.Tanh()
        # )
        # cond_embedding_size = Config.topic_embed_size + 4 + 4 + Config.ctx_hidden_size

        # # ------------------recognitionNetwork------------------
        # recog_input_size = cond_embedding_size + output_embedding_size
        # if self.use_hcf:
        #     recog_input_size += 30
        # self.recogNet_mulogvar = nn.Linear(recog_input_size, Config.latent_size * 2)

        # # ------------------priorNetwork------------------
        # self.priorNet_mulogvar = nn.Sequential(
        #     nn.Linear(cond_embedding_size, np.maximum(Config.latent_size * 2, 100)),
        #     nn.Tanh(),
        #     nn.Linear(np.maximum(Config.latent_size * 2, 100), Config.latent_size * 2)
        # )
        # gen_inputs_size = cond_embedding_size + Config.latent_size

        # # BOW loss
        # self.bow_project = nn.Sequential(
        #     nn.Linear(gen_inputs_size, 400),
        #     nn.Tanh(),
        #     nn.Dropout(1 - config.keep_prob),
        #     nn.Linear(400, self.vocab_size))

        # # Y loss
        # if self.use_hcf:
        #     self.da_project = nn.Sequential(
        #         nn.Linear(gen_inputs_size, 400),
        #         nn.Tanh(),
        #         nn.Dropout(1 - config.keep_prob),
        #         nn.Linear(400, self.da_vocab_size))
        #     dec_inputs_size = gen_inputs_size + 30
        # else:
        #     dec_inputs_size = gen_inputs_size

        # # Decoder
        # if config.num_layer > 1:
        #     self.dec_init_state_net = nn.ModuleList([nn.Linear(dec_inputs_size, self.dec_cell_size) for i in range(config.num_layer)])
        # else:
        #     self.dec_init_state_net = nn.Linear(dec_inputs_size, self.dec_cell_size)

        # # decoder
        # dec_input_embedding_size = self.embed_size
        # if self.use_hcf:
        #     dec_input_embedding_size += 30
        # self.dec_cell = self.get_rnncell(config.cell_type, dec_input_embedding_size, self.dec_cell_size, config.keep_prob, config.num_layer)
        # self.dec_cell_proj = nn.Linear(self.dec_cell_size, self.vocab_size)


    def forward(self, feed_dict, mode='train'):
        for k, v in feed_dict.items():
            setattr(self, k, v)

        max_dialog_len = self.input_contexts.size(1)
        max_out_len = self.output_tokens.size(1)
        batch_size = self.input_contexts.size(0)

        # ============================ embedding ===========================
        t_embedding = self.topic_embedding(self.topics)
        if Config.use_hcf:
            d_embedding = self.act_embedding(self.output_des)
        input_embedding = self.word_embedding(self.input_contexts.view(-1, Config.max_utt_length))
        output_embedding = self.word_embedding(self.output_tokens)

        # ============================ utt encoder ============================
        # in : torch.Size([540, 40, 200])
        # out: torch.Size([540, 600])
        input_embedding, sent_size = self.utt_encoder(input_embedding)
        output_embedding, _ = self.utt_encoder(output_embedding, self.output_lens)

        # reshape input into dialogs
        input_embedding = input_embedding.view(-1, max_dialog_len, sent_size)

        # ============================ ctx encoder ============================
        # convert floors into 1 hot
        floor_one_hot = self.floors.new_zeros((self.floors.numel(), 2), dtype=torch.float)
        floor_one_hot.data.scatter_(1, self.floors.view(-1,1), 1)
        floor_one_hot = floor_one_hot.view(-1, max_dialog_len, 2)

        # input of Context RNN
        joint_embedding = torch.cat([input_embedding, floor_one_hot], 2)
        _, ctx_enc_last_state = self.ctx_encoder(joint_embedding, self.context_lens)

        if Config.ctx_num_layer > 1:
            ctx_enc_last_state = torch.cat([_ for _ in torch.unbind(ctx_enc_last_state)], 1)
        else:
            ctx_enc_last_state = ctx_enc_last_state.squeeze(0)

        # ============================ cvae ============================
        # combine with other attributes
        if Config.use_hcf:
            attribute_embedding = d_embedding
            attribute_fc1 = self.attribute_fc1(attribute_embedding)

        cond_list = [t_embedding, self.my_profile, self.ot_profile, ctx_enc_last_state]
        cond_embedding = torch.cat(cond_list, 1)

        if Config.use_hcf:
            recog_input = torch.cat([cond_embedding, output_embedding, attribute_fc1], 1)
        else:
            recog_input = torch.cat([cond_embedding, output_embedding], 1)

        recog_mu, recog_logvar = self.recogNet_mulogvar(recog_input)
        prior_mu, prior_logvar = self.priorNet_mulogvar(cond_embedding)

        # use sampled Z or posterior Z
        if self.use_prior:
            latent_sample = sample_gaussian(prior_mu, prior_logvar)
        else:
            latent_sample = sample_gaussian(recog_mu, recog_logvar)

        # ============================ bow and y project ============================
        gen_inputs = torch.cat([cond_embedding, latent_sample], 1)
        bow_logits = self.bow_project(gen_inputs)

        if Config.use_hcf:
            y_logits = self.y_project(gen_inputs)
            y_prob = F.softmax(y_logits, dim=1)
            pred_attribute_embedding = torch.matmul(y_prob, self.act_embedding.weight)

            if mode == 'test':
                selected_attribute_embedding = pred_attribute_embedding
            else:
                selected_attribute_embedding = attribute_embedding
            enc_ouputs = torch.cat([gen_inputs, selected_attribute_embedding], 1)
        else:
            y_logits = gen_inputs.new_zeros(batch_size, Config.act_vocab_size)
            enc_ouputs = gen_inputs

        # ============================ decoder ============================
        dec_input_tokens = self.output_tokens[:, :-1]
        dec_input_embedding = self.word_embedding(dec_input_tokens)
        dec_seq_lens = self.output_lens - 1
        dec_outs, _, final_context_state = self.decoder(dec_input_embedding, enc_ouputs, dec_seq_lens, selected_attribute_embedding)

        if final_context_state is not None:
            dec_out_words = final_context_state
        else:
            dec_out_words = torch.max(dec_outs, 2)[1]

        return dec_out_words


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")
    train_feed = Data(corpus_file, "train")

    model = KgRnnCVAE()
    if Config.use_gpu:
        model.cuda()

    train_feed.epoch_init()
    
    while True:
        batch_data = train_feed.next_batch()
        if batch_data is None:
            break
        feed_dict = model.batch_feed(batch_data, use_prior=False)
        model(feed_dict, mode='train')
        exit()


