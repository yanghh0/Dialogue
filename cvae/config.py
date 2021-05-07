
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