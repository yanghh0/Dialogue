
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
from utils.functions import gaussian_kld, get_bleu_stats

import tensorboardX as tb
import tensorboardX.summary
import tensorboardX.writer


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class Chatbot:
    def __init__(self, corpus_file):
        self.train_feed = Data(corpus_file, "train")
        self.valid_feed = Data(corpus_file, "valid")
        self.test_feed = Data(corpus_file, "test")
        self.model = KgRnnCVAE()

        self.model.apply(self.weight_init)
        self.model.word_embedding.weight.data.copy_(torch.from_numpy(np.array(Config.word2vec)))
        self.model.word_embedding.weight.data[0].fill_(0)

        self.optimizer = optim.Adam(self.model.parameters(), Config.init_lr)
        self.train_summary_writer = tb.writer.FileWriter("checkpoint")

        self.global_t = 1
        self.forward_only = False
        self.patience = 10
        self.dev_loss_threshold = np.inf
        self.best_dev_loss = np.inf

        if Config.use_gpu:
            self.model.cuda()

    @staticmethod
    def weight_init(m):
        for p in m.parameters():
            nn.init.uniform_(p.data, -1.0 * Config.init_w, Config.init_w)

    def cal_loss(self, mode,
                 dec_outs, output_des,
                 labels,
                 bow_logits, act_logits,
                 recog_mu, recog_logvar, prior_mu, prior_logvar):
        label_mask = torch.sign(labels).detach().float()

        rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)),
                                  labels.reshape(-1), reduction='none').view(dec_outs.size()[:-1])
        rc_loss = torch.sum(rc_loss * label_mask, 1)
        avg_rc_loss = rc_loss.mean()

        # used only for perpliexty calculation. Not used for optimzation
        rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

        bow_loss = -F.log_softmax(bow_logits, dim=1).gather(1, labels) * label_mask
        bow_loss = torch.sum(bow_loss, 1)
        avg_bow_loss  = torch.mean(bow_loss)

        if Config.use_hcf:
            avg_act_loss = F.cross_entropy(act_logits, output_des)
        else:
            avg_act_loss = avg_bow_loss.new_tensor(0)

        kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
        avg_kld = torch.mean(kld)

        if mode == 'train':
            kl_weights = min(self.global_t / Config.full_kl_step, 1.0)
        else:
            kl_weights = 1.0

        kl_w = kl_weights
        elbo = avg_rc_loss + kl_weights * avg_kld
        aug_elbo = avg_bow_loss + avg_act_loss + elbo

        return aug_elbo, \
               elbo.item(), avg_bow_loss.item(), rc_ppl.item(), avg_rc_loss.item(), avg_kld.item(), \
               kl_w

    def train_epoch(self):
        elbo_losses = []
        bow_losses = []
        rc_ppls = []
        rc_losses = []
        kl_losses = []

        local_t = 0
        start_time = time.time()
        loss_names = ["elbo_loss", "bow_loss", "rc_peplexity", "rc_loss", "kl_loss"]

        while True:
            batch_data = self.train_feed.next_batch()
            if batch_data is None:
                break

            feed_dict = self.model.batch_feed(batch_data, use_prior=False)
            info, _ = self.model(feed_dict, mode='train')
            aug_elbo, elbo_loss, bow_loss, rc_ppl, rc_loss, kl_loss, kl_w = self.cal_loss(mode="train", *info)

            self.optimizer.zero_grad()
            aug_elbo.backward()
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)
            self.optimizer.step()
            self.global_t += 1
            local_t += 1

            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            if local_t % (self.train_feed.num_batch // 10) == 0:
                print_loss("%.2f" % (self.train_feed.pointer / float(self.train_feed.num_batch)),
                           loss_names, 
                           [elbo_losses, bow_losses, rc_ppls, rc_losses, kl_losses], 
                           "kl_w %f" % kl_w)

        # finish epoch!
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = print_loss("Epoch Done", loss_names, 
                                [elbo_losses, bow_losses, rc_ppls, rc_losses, kl_losses],
                                "step time %.4f" % (epoch_time / self.train_feed.num_batch))
        return avg_losses[0]

    def valid_epoch(self):
        elbo_losses = []
        bow_losses = []
        rc_ppls = []
        rc_losses = []
        kl_losses = []

        while True:
            batch_data = self.valid_feed.next_batch()
            if batch_data is None:
                break

            feed_dict = self.model.batch_feed(batch_data, use_prior=False)
            with torch.no_grad():
                info, _ = self.model(feed_dict, mode='valid')
            aug_elbo, elbo_loss, bow_loss, rc_ppl, rc_loss, kl_loss, kl_w = self.cal_loss(mode="valid", *info)

            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

        avg_losses = print_loss("ELBO_VALID", 
                                ["elbo_loss", "bow_loss", "rc_peplexity", "rc_loss", "kl_loss"],
                                [elbo_losses, bow_losses, rc_ppls, rc_losses, kl_losses], "")
        return avg_losses[0]

    def test_epoch(self, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []
        repeat = 5

        while True:
            batch_data = self.test_feed.next_batch()
            if batch_data is None:
                break

            feed_dict = self.model.batch_feed(batch_data, use_prior=True, repeat=repeat)
            with torch.no_grad():
                _, _, _, bow_logits, act_logits, _, _, _, _, dec_out_words = self.model(feed_dict, mode='test')

            word_outs = dec_out_words.cpu().numpy()
            act_logits = act_logits.cpu().numpy()
            sample_words = np.split(word_outs, repeat, axis=0)
            sample_des = np.split(act_logits, repeat, axis=0)

            true_srcs = feed_dict["input_contexts"].cpu().numpy()
            true_floor = feed_dict["floors"].cpu().numpy()
            true_src_lens = feed_dict["context_lens"].cpu().numpy()
            true_topics = feed_dict["topics"].cpu().numpy()
            true_outs = feed_dict["output_tokens"].cpu().numpy()
            true_des = feed_dict["output_des"].cpu().numpy()
            local_t += 1

            word_vocab = self.test_feed.word_alphabet.index2token
            topic_vocab = self.test_feed.topic_alphabet.index2token
            act_vocab = self.test_feed.act_alphabet.index2token

            for b_id in range(self.test_feed.batch_size):
                dest.write("Batch %d index %d of topic %s\n" % (local_t, b_id, topic_vocab[true_topics[b_id]]))
                start = np.maximum(0, true_src_lens[b_id] - 5)

                for t_id in range(start, true_srcs.shape[1], 1):
                    src_str = " ".join([word_vocab[e] for e in true_srcs[b_id, t_id].tolist() if e != 0])
                    dest.write("Src %d-%d: %s\n" % (t_id, true_floor[b_id, t_id], src_str))

                true_tokens = [word_vocab[e] for e in true_outs[b_id].tolist() if e not in [0, SOS_token, EOS_token]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                act_str = act_vocab[true_des[b_id]]
                dest.write("Target (%s) >> %s\n" % (act_str, true_str))

                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_act = np.argmax(sample_des[r_id], axis=1)[0]
                    pred_tokens = [word_vocab[e] for e in pred_outs[b_id].tolist() if e != EOS_token and e != 0]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d (%s) >> %s\n" % (r_id, act_vocab[pred_act], pred_str))
                    local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)
                dest.write("\n")

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2*(avg_prec_bleu*avg_recall_bleu) / (avg_prec_bleu+avg_recall_bleu+10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")
        print("Done testing")

    def train_model(self):
        if not self.forward_only:
            dm_checkpoint_path = os.path.join("checkpoint", self.model.__class__.__name__+ "-%d.pth")
            for epoch in range(Config.max_epoch):
                self.model.train()
                self.train_feed.epoch_init()
                train_loss = self.train_epoch()

                self.model.eval()
                self.valid_feed.epoch_init()
                valid_loss = self.valid_epoch()

                done_epoch = epoch + 1
                if valid_loss < self.best_dev_loss:
                    print("Save model!!")
                    torch.save(self.model.state_dict(), dm_checkpoint_path %(epoch))
                    self.best_dev_loss = valid_loss
        else:
            self.model.eval()


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")

    cbt = Chatbot(corpus_file)
    cbt.train_model()