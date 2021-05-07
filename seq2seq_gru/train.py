
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import torch.nn as nn
import tqdm
import random
from torch import optim
import torch.nn.functional as F
from seq2seq_gru.seq2seq import EncoderRNN, LuongAttnDecoderRNN
from seq2seq_gru.config import Config
from utils.cornell_movie_data import Data


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class QATrainer:
    def __init__(self, data_obj, load_model_name=None):
        self.data_obj = data_obj
        self.vocab_size = data_obj.word_alphabet.num_tokens
        self.load_model_name = load_model_name

        # network
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=Config.hidden_size)
        self.encoder = EncoderRNN(hidden_size=Config.hidden_size, 
                                  embedding=self.embedding, 
                                  n_layers=Config.encoder_n_layers, 
                                  dropout=Config.dropout)
        self.decoder = LuongAttnDecoderRNN(embedding=self.embedding, 
                                           hidden_size=Config.hidden_size, 
                                           vocab_size=self.vocab_size, 
                                           n_layers=Config.decoder_n_layers, 
                                           dropout=Config.dropout)

        # optimizer
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=Config.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=Config.learning_rate*Config.decoder_learning_ratio)

        if self.load_model_name:
            checkpoint = torch.load(self.load_model_name)
            self.embedding.load_state_dict(checkpoint['embedding'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

        if Config.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

            if self.load_model_name:
                for state in self.encoder_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

                for state in self.decoder_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        print('Models built and ready to go!')

    def maskNLLLoss(self, output, target, mask):
        """shape of output: (b, v)
           shape of target: (b,)
           shape of mask: (b,)
        """
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(output, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        return loss, nTotal.item() 

    def train_batch(self, inputs, inputs_length, targets, masks, max_target_len):
        """shape of inputs: (t, b)
           shape of targets: (t, b)
        """
        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        if Config.use_gpu:
            inputs = inputs.to(Config.device)
            targets = targets.to(Config.device)
            masks = masks.to(Config.device)

        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(inputs, inputs_length)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(Config.batch_size)]])
        if Config.use_gpu:
            decoder_input = decoder_input.to(Config.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:Config.decoder_n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < Config.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Teacher forcing: next input is current target
                decoder_input = targets[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, targets[t], masks[t])
                loss += mask_loss
                n_totals += nTotal
                print_losses.append(mask_loss.item() * nTotal)
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(Config.batch_size)]])
                if Config.use_gpu:
                    decoder_input = decoder_input.to(Config.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, targets[t], masks[t])
                loss += mask_loss
                n_totals += nTotal
                print_losses.append(mask_loss.item() * nTotal)

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), Config.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), Config.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def train_epoch(self, epoch_th):
        print_loss = 0
        iteration = 0
        epoch_loss = 0
        while True:
            batch_data = self.data_obj.next_batch()
            if batch_data is None:
                break 
            loss = self.train_batch(*batch_data)
            print_loss += loss
            epoch_loss += loss
            iteration += 1
            if iteration % Config.print_every == 0:
                print_loss_avg = print_loss / Config.print_every
                print("Percent complete in {} epoch: {}/{}; Average loss: {:.4f}".format(
                       epoch_th, iteration, self.data_obj.num_batch, print_loss_avg))
                print_loss = 0
        return epoch_loss / iteration
            
    def train_model(self):
        print("Starting Training!")
        
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        min_loss = 100000.0

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, Config.epochs + 1):
            self.data_obj.epoch_init(Config.batch_size)
            loss = self.train_epoch(iteration)

            # Save checkpoint
            if loss < min_loss:
                min_loss = loss
                model_name = 'model_loss_{:.3f}.chkpt'.format(loss)
                checkpoint = {
                    'embedding': self.embedding.state_dict(),
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
    trainer.train_model()
