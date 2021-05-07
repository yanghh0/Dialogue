

import torch


class Config():
	hidden_size = 500
	encoder_n_layers = 2
	decoder_n_layers = 2
	dropout = 0.1
	batch_size = 64
	learning_rate = 0.0001
	decoder_learning_ratio = 5.0
	clip = 50.0
	teacher_forcing_ratio = 1.0
	epochs = 10
	print_every = 1
	use_gpu = True
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	beam_size = 2
	max_sentence_length = 10