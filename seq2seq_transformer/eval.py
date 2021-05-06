
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cornell_movie_data import Data
from utils.functions import normalizeString
from seq2seq_transformer.mask import get_pad_mask, get_subsequent_mask
from seq2seq_transformer.seq2seq import Encoder, Decoder


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class QAEvaluator:
    """Load a trained model and generate in gready/beam search fashion.
    """
    def __init__(self, data_obj, load_model_name, use_gpu=False, embed_sharing=True):
        # parameter of network
        self.load_model_name = load_model_name
        self.data_obj = data_obj
        self.vocab_size = data_obj.word_alphabet.num_tokens
        self.d_model = 512
        self.d_inner = 2048
        self.n_head = 8
        self.n_layers = 6
        self.beam_size = 3
        self.max_length = 20
        self.beam_space = torch.full((self.beam_size, self.max_length), PAD_token, dtype=torch.long)
        self.beam_space[:, 0] = SOS_token
        self.len_map = torch.arange(1, self.max_length + 1, dtype=torch.long).unsqueeze(0)
        self.alpha = 0.7

        # network
        self.encoder = Encoder(self.vocab_size, self.d_model, self.d_inner, self.n_head, self.n_layers)
        self.decoder = Decoder(self.vocab_size, self.d_model, self.d_inner, self.n_head, self.n_layers)
        if embed_sharing:
            self.encoder.embedding.weight = self.decoder.embedding.weight

        # Load network parameters
        checkpoint = torch.load(self.load_model_name)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

        print('Trained model state loaded.')

    def _get_init_state(self, src_seq):
        """shape of src_seq: (b=1, t)
        """

        # Initialize decoder input with SOS_token
        init_seq = torch.LongTensor([[SOS_token]])

        # Get mask
        src_mask = get_pad_mask(src_seq, PAD_token)
        trg_mask = get_subsequent_mask(init_seq)

        # Forward input through encoder model
        # shape of encoder_output: (b=1, t, h)
        # shape of decoder_output: (b=1, t=1, v)
        encoder_output = self.encoder(src_seq, src_mask)
        decoder_output = self.decoder(init_seq, trg_mask, encoder_output, src_mask)
        decoder_output = F.softmax(decoder_output, dim=-1)

        # shape of best_k_probs, best_k_idx: (b=1, beam_size)
        best_k_probs, best_k_idx = decoder_output[:, -1, :].topk(self.beam_size)

        # shape of scores: (beam_size,)
        scores = torch.log(best_k_probs).view(self.beam_size)
        self.beam_space[:, 1] = best_k_idx[0]

        encoder_output = encoder_output.repeat(self.beam_size, 1, 1)
        return encoder_output, scores


    def eval(self, src_seq):
        """shape of src_seq: (t, b=1)
        """
        src_seq = src_seq.t()
        assert src_seq.size(0) == 1

        # Ensure dropout layers are in eval mode
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            encoder_output, scores = self._get_init_state(src_seq)

            ans_idx = 0   # default
            for step in range(2, self.max_length):    # decode up to max length
                trg_seq = self.beam_space[:, :step]

                # Get mask 
                src_mask = get_pad_mask(src_seq, PAD_token)
                trg_mask = get_subsequent_mask(trg_seq)

                decoder_output = self.decoder(trg_seq, trg_mask, encoder_output, src_mask)
                decoder_output = F.softmax(decoder_output, dim=-1)

                best_k2_probs, best_k2_idx = decoder_output[:, -1, :].topk(self.beam_size)
                scores = torch.log(best_k2_probs).view(self.beam_size, -1) + scores.view(self.beam_size, 1)

                scores, best_k_idx_in_k2 = scores.view(-1).topk(self.beam_size)

                best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // self.beam_size, best_k_idx_in_k2 % self.beam_size
                best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

                self.beam_space[:, :step] = self.beam_space[best_k_r_idxs, :step]
                self.beam_space[:, step] = best_k_idx

                eos_locs = self.beam_space == EOS_token   

                # self.len_map will broadcast to shape: (beam_size, max_length)
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, self.max_length).min(1)

                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** self.alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return self.beam_space[ans_idx][:seq_lens[ans_idx]]


if __name__ == "__main__":
    corpus_name = "cornell movie-dialogs corpus"
    datafile = os.path.join("..", "datasets", corpus_name, "formatted_movie_lines.txt")
    model_name = os.path.join("checkpoint", "model_loss_3.269.chkpt")

    data_obj = Data()
    data_obj.build_alphabet(datafile)
    data_obj.trimRareWords()

    evaluator = QAEvaluator(data_obj, model_name)

    input_sentence = ''
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break

            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            print("Q: %s" % input_sentence)

            # words -> indexes
            indexes_inputs = [data_obj.word_alphabet.sequence_to_indexes(input_sentence)]

            # Transpose dimensions of batch to match models' expectations
            indexes_inputs = torch.LongTensor(indexes_inputs).transpose(0, 1)

            # Decode sentence with evaluator
            tokens = evaluator.eval(indexes_inputs)

            # indexes -> words
            decoded_words = [data_obj.word_alphabet.index2token[token.item()] for token in tokens]
            decoded_words = ' '.join(decoded_words)
            decoded_words = decoded_words.replace('SOS', '').replace('EOS', '')

            print("A: %s" % decoded_words.strip())
        except KeyError:
            print("Error: Encountered unknown word.")