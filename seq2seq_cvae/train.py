
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
import tensorboardX as tb
import tensorboardX.summary
import tensorboardX.writer
from seq2seq_cvae.kg_cvae import KgRnnCVAE
from seq2seq_cvae.config import Config
from utils.Sw_data import Data
from utils.functions import print_loss 


class ChatbotTrainer:
    def __init__(self, data_obj):
        self.train_feed = data_obj
        self.model = KgRnnCVAE()
        self.train_summary_writer = tb.writer.FileWriter("checkpoint")

        self.model.apply(lambda m: [torch.nn.init.uniform_(p.data, -1.0 * Config.init_w, Config.init_w) 
            for p in self.model.parameters()])
        self.model.word_embedding.weight.data.copy_(torch.from_numpy(np.array(Config.word2vec)))

        self.optimizer = optim.Adam(self.model.parameters(), Config.init_lr)

        if Config.use_gpu:
            self.model.cuda()

    def train_epoch(self, global_t):
        self.train_feed.epoch_init()

        elbo_losses = []
        bow_losses = []
        rc_losses = []
        kl_losses = []
        local_t = 0
        start_time = time.time()
        loss_names =  ["elbo_loss", "bow_loss", "rc_loss", "kl_loss"]

        while True:
            batch_data = train_feed.next_batch()
            if batch_data is None:
                break
            feed_dict = self.model.batch_feed(batch_data, global_t, use_prior=False)
            self.model(feed_dict, mode='train')
            aug_elbo, elbo, avg_bow_loss, avg_rc_loss, avg_kld, avg_act_loss, kl_w = self.model(feed_dict, mode='train')

            self.optimizer.zero_grad()
            aug_elbo.backward()
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)
            self.optimizer.step()

            elbo_losses.append(elbo)
            bow_losses.append(avg_bow_loss)
            rc_losses.append(avg_rc_loss)
            kl_losses.append(avg_kld)

            summary_op = [
                tb.summary.scalar("model/loss/act_loss", avg_act_loss),
                tb.summary.scalar("model/loss/rc_loss", avg_rc_loss),
                tb.summary.scalar("model/loss/elbo", elbo),
                tb.summary.scalar("model/loss/kld", avg_kld),
                tb.summary.scalar("model/loss/bow_loss", avg_bow_loss)
            ]

            for summary in summary_op:
                self.train_summary_writer.add_summary(summary, global_t)

            global_t += 1
            local_t += 1

            if local_t % (train_feed.num_batch // 10) == 0:
                print_loss("%.2f" % (train_feed.pointer / float(train_feed.num_batch)),
                           loss_names,
                           [elbo_losses, bow_losses, rc_losses, kl_losses],
                           "kl_w %f" % kl_w)

        # finish epoch!
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = print_loss("Epoch Done", loss_names,
                                [elbo_losses, bow_losses, rc_losses, kl_losses],
                                "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def train_model(self):
        print("Starting Training!")

        self.model.train() 

        global_t = 1
        for epoch in range(Config.max_epoch):
            global_t, train_loss = self.train_epoch(global_t)


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")
    train_feed = Data(corpus_file, "train")

    cbt = ChatbotTrainer(train_feed)
    cbt.train_model()