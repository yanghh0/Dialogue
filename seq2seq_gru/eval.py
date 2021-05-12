
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import torch.nn as nn
from seq2seq_gru.seq2seq import EncoderRNN, LuongAttnDecoderRNN
from seq2seq_gru.config import Config
from utils.functions import normalizeString
from utils.cornell_movie_data import Data


# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class QAEvaluator:
    """Load a trained model and generate in gready/beam search fashion.
    """
    def __init__(self, data_obj, load_model_name):
        self.load_model_name = load_model_name
        self.data_obj = data_obj
        self.vocab_size = data_obj.word_alphabet.num_tokens

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

        # Load network parameters
        checkpoint = torch.load(self.load_model_name)
        self.embedding.load_state_dict(checkpoint['embedding'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

        print('Trained model state loaded.')

    def eval(self, inputs, inputs_length, max_length):
        """shape of inputs: (t, b=1)
        """

        # Ensure dropout layers are in eval mode
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():

            # Forward input through encoder model
            encoder_outputs, encoder_hidden = self.encoder(inputs, inputs_length)

            # Prepare encoder's final hidden layer to be first hidden input to the decoder
            decoder_hidden = encoder_hidden[:Config.decoder_n_layers]

            # Initialize decoder input with SOS_token
            decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_token

            # Initialize tensors to append decoded words to
            all_tokens = torch.zeros([0], dtype=torch.long)
            all_scores = torch.zeros([0])

            # Iteratively decode one word token at a time
            for _ in range(max_length):
                # Forward pass through decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                # Record token and score
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input = torch.unsqueeze(decoder_input, 0)
                if decoder_input.item() == EOS_token:
                    break

        # Return collections of word tokens and scores
        return all_tokens, all_scores


if __name__ == "__main__":
    corpus_name = "cornell movie-dialogs corpus"
    datafile = os.path.join("..", "datasets", corpus_name, "formatted_movie_lines.txt")
    model_name = os.path.join("checkpoint", "model_loss_2.462.chkpt")

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

            # Create lengths tensor
            input_lengths = torch.tensor([len(indexes) for indexes in indexes_inputs])

            # Transpose dimensions of batch to match models' expectations
            indexes_inputs = torch.LongTensor(indexes_inputs).transpose(0, 1)

            # Decode sentence with evaluator
            tokens, scores = evaluator.eval(indexes_inputs, input_lengths, max_length=Config.max_sentence_length)

            # indexes -> words
            decoded_words = [data_obj.word_alphabet.index2token[token.item()] for token in tokens]

            # Format and print response sentence
            decoded_words[:] = [x for x in decoded_words if not (x == 'EOS' or x == 'PAD')]
            print("A: %s" % ' '.join(decoded_words))
        except KeyError:
            print("Error: Encountered unknown word.")