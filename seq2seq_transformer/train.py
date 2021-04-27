
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.data import Data
from seq2seq_transformer.mask import get_pad_mask, get_subsequent_mask
from seq2seq_transformer.seq2seq import Encoder, Decoder


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class QATrainer:
    def __init__(self, data_obj, use_gpu=True, embed_sharing=True):
        # parameter of network
        self.data_obj = data_obj
        self.vocab_size = data_obj.word_alphabet.num_words
        self.d_model = 512
        self.d_inner = 2048
        self.n_head = 8
        self.n_layers = 6
        self.batch_size = 64

        # parameter of training
        self.learning_rate = 0.0001
        self.n_iteration = 5000
        self.save_every = 500
        self.print_every = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = use_gpu

        # network
        self.encoder = Encoder(self.vocab_size, self.d_model, self.d_inner, self.n_head, self.n_layers)
        self.decoder = Decoder(self.vocab_size, self.d_model, self.d_inner, self.n_head, self.n_layers)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        if embed_sharing:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        # optimizer
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

        if use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        print('Models built and ready to go!')

    def cal_loss(self, dec_outputs, ground_truth):
        dec_outputs = dec_outputs.view(-1, dec_outputs.size(2))
        ground_truth = ground_truth.contiguous().view(-1)
        loss = F.cross_entropy(dec_outputs, ground_truth, ignore_index=PAD_token, reduction='mean')
        non_pad_mask = ground_truth.ne(PAD_token)
        nTotal = non_pad_mask.sum().item()
        return loss, nTotal

    def train(self, inputs, targets):
        """shape of inputs: (t, b)
           shape of inputs: (t', b)
        """

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        inputs = inputs.transpose(0, 1)
        targets = targets.transpose(0, 1)
        targets = torch.LongTensor([[SOS_token] + sentence for sentence in targets.numpy().tolist()])
        targets, ground_truth = targets[:, :-1], targets[:, 1:]

        src_mask = get_pad_mask(inputs, PAD_token)
        trg_mask = get_pad_mask(targets, PAD_token) & get_subsequent_mask(targets)

        if self.use_gpu:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            ground_truth = ground_truth.to(self.device)
            src_mask = src_mask.to(self.device)
            trg_mask = trg_mask.to(self.device)

        enc_outputs = self.encoder(inputs, src_mask)
        dec_outputs = self.decoder(targets, trg_mask, enc_outputs, src_mask)

        loss, nTotal = self.cal_loss(dec_outputs, ground_truth)

        # Perform backpropatation
        loss.backward()

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def trainIters(self):
        print("Starting Training!")
        
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

        # Prepare data
        yield_training_batch = self.data_obj.data_generator(self.batch_size)

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, self.n_iteration + 1):
            # Extract fields from batch
            inputs, _, targets, _, _ = next(yield_training_batch)

            # Run a training iteration with batch
            loss = self.train(inputs, targets)
            print_loss += loss

            # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / self.n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if iteration % self.save_every == 0:
                model_name = 'model_loss_{:.3f}.chkpt'.format(loss)
                checkpoint = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join("checkpoint", model_name))


if __name__ == "__main__":
    corpus_name = "cornell movie-dialogs corpus"
    datafile = os.path.join("..", "datasets", corpus_name, "formatted_movie_lines.txt")

    data_obj = Data()
    data_obj.build_alphabet(datafile)
    data_obj.trimRareWords()

    trainer = QATrainer(data_obj)
    trainer.trainIters()