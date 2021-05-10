
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
from utils.Sw_data import Data
from utils.functions import print_loss 
from seq2seq_cvae.kg_cvae import KgRnnCVAE
from seq2seq_cvae.config import Config
from utils.functions import gaussian_kld


class ChatbotTrainer:
    def __init__(self, data_obj):
        self.train_feed = data_obj
        self.model = KgRnnCVAE()

        self.model.apply(lambda m: [torch.nn.init.uniform_(p.data, -1.0 * Config.init_w, Config.init_w) 
            for p in m.parameters()])
        self.model.word_embedding.weight.data.copy_(torch.from_numpy(np.array(Config.word2vec)))

        self.optimizer = optim.Adam(self.model.parameters(), Config.init_lr)
        self.global_t = 1

        self.elbo_losses = []
        self.bow_losses = []
        self.rc_losses = []
        self.kl_losses = []
        self.losses = [self.elbo_losses, self.bow_losses, self.rc_losses, self.kl_losses]

        if Config.use_gpu:
            self.model.cuda()

    def cal_loss(self, dec_outs, output_des, 
                 labels, 
                 bow_logits, act_logits,
                 recog_mu, recog_logvar, prior_mu, prior_logvar):
        label_mask = torch.sign(labels).detach().float()
        rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)), labels.reshape(-1), reduction='none').view(dec_outs.size()[:-1])

        rc_loss = torch.sum(rc_loss * label_mask, 1)
        avg_rc_loss = rc_loss.mean()

        bow_loss = -F.log_softmax(bow_logits, dim=1).gather(1, labels) * label_mask
        bow_loss = torch.sum(bow_loss, 1)
        avg_bow_loss  = torch.mean(bow_loss)

        if Config.use_hcf:
            avg_act_loss = F.cross_entropy(act_logits, output_des)
        else:
            avg_act_loss = avg_bow_loss.new_tensor(0)

        kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
        avg_kld = torch.mean(kld)
        kl_weights = min(self.global_t / Config.full_kl_step, 1.0)

        kl_w = kl_weights
        elbo = avg_rc_loss + kl_weights * avg_kld
        aug_elbo = avg_bow_loss + avg_act_loss + elbo

        self.elbo_losses.append(elbo.item())
        self.bow_losses.append(avg_bow_loss.item())
        self.rc_losses.append(avg_rc_loss.item())
        self.kl_losses.append(avg_kld.item())

        return aug_elbo, kl_w

    def train_epoch(self):
        self.elbo_losses = []
        self.bow_losses = []
        self.rc_losses = []
        self.kl_losses = []

        local_t = 0
        start_time = time.time()
        loss_names =  ["elbo_loss", "bow_loss", "rc_loss", "kl_loss"]

        while True:
            batch_data = train_feed.next_batch()
            if batch_data is None:
                break
            feed_dict = self.model.batch_feed(batch_data, use_prior=False)
            self.model(feed_dict, mode='train')
            info = self.model(feed_dict, mode='train')
            aug_elbo, kl_w = self.cal_loss(*info)

            self.optimizer.zero_grad()
            aug_elbo.backward()
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)
            self.optimizer.step()
            self.global_t += 1
            local_t += 1

            if local_t % (train_feed.num_batch // 10) == 0:
                print_loss("%.2f" % (train_feed.pointer / float(train_feed.num_batch)), loss_names, self.losses, "kl_w %f" % kl_w)

        # finish epoch!
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = print_loss("Epoch Done", loss_names, self.losses, "step time %.4f" % (epoch_time / train_feed.num_batch))

        return avg_losses[0]

    def train_model(self):
        print("Starting Training!")

        self.model.train() 
        for epoch in range(Config.max_epoch):
            self.train_feed.epoch_init()
            loss = self.train_epoch()


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")
    train_feed = Data(corpus_file, "train")

    cbt = ChatbotTrainer(train_feed)
    cbt.train_model()