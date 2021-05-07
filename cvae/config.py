
import torch


class Config():
    max_utt_length = 40       # max number of words in an utterance
    word_embed_size = 200     # word embedding size
    topic_embed_size = 30     # topic embedding size
    act_embed_size = 30       # dialog act embedding size
    use_hcf = True            # use dialog act in training (if turn off kgCVAE -> CVAE)
    step_size = 1             # internal usage
    batch_size = 60           # mini-batch size
    backward_size = 10        # how many utterance kept in the context window
    ctx_hidden_size = 600     # context encoder hidden size
    utt_hidden_size = 300     # utterance encoder hidden size
    res_hidden_size = 400     # response decoder hidden size
    latent_size = 200         # the dimension of latent variable
    num_layer = 1             # number of context RNN layers