
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
from utils.cornell_movie_data import Data


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class QATrainer:
    def __init__(self, data_obj, load_model_name=None, use_gpu=True):
        # parameter of network
        self.data_obj = data_obj
        self.vocab_size = data_obj.word_alphabet.num_tokens
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 128

        # Load model if a loadFilename is provided
        self.load_model_name = load_model_name

        # parameter of training
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.n_iteration = 4000
        self.save_every = 500
        self.print_every = 1
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # network
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size)
        self.encoder = EncoderRNN(hidden_size=self.hidden_size, 
                                  embedding=self.embedding, 
                                  n_layers=self.encoder_n_layers, 
                                  dropout=self.dropout)
        self.decoder = LuongAttnDecoderRNN(embedding=self.embedding, 
                                           hidden_size=self.hidden_size, 
                                           vocab_size=self.vocab_size, 
                                           n_layers=self.decoder_n_layers, 
                                           dropout=self.dropout)

        # optimizer
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate*self.decoder_learning_ratio)

        if self.load_model_name:
            checkpoint = torch.load(self.load_model_name)
            self.embedding.load_state_dict(checkpoint['embedding'])
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

        if use_gpu:
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

    def train(self, inputs, inputs_length, targets, masks, max_target_len):
        """shape of inputs: (t, b)
           shape of targets: (t, b)
        """
        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        if self.use_gpu:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(inputs, inputs_length)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(self.batch_size)]])
        if self.use_gpu:
            decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder_n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

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
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                if self.use_gpu:
                    decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, targets[t], masks[t])
                loss += mask_loss
                n_totals += nTotal
                print_losses.append(mask_loss.item() * nTotal)

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals

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

        if self.load_model_name:
            checkpoint = torch.load(self.load_model_name)
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, self.n_iteration + 1):
            # Extract fields from batch
            inputs, inputs_lengths, outputs, masks, max_target_len = next(yield_training_batch)

            # Run a training iteration with batch
            loss = self.train(inputs, inputs_lengths, outputs, masks, max_target_len)
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
                    'iteration': iteration,
                    'embedding': self.embedding.state_dict(),
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                    'loss': loss
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
