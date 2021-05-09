
import torch


class Config:
    max_utt_length = 40

    # use dialog act in training (if turn off kgCVAE -> CVAE)
    use_hcf = True
    use_gpu = True

    # It needs to be set externally. The current value is set arbitrarily.
    word_vocab_size = 1
    topic_vocab_size = 1
    act_vocab_size = 1

    word_embed_size = 200
    topic_embed_size = 30
    act_embed_size = 30

    utt_num_layer = 1
    utt_input_size = word_embed_size
    utt_hidden_size = 300

    ctx_num_layer = 1
    ctx_input_size = utt_hidden_size * 2 + 2
    ctx_hidden_size = 600

    prior_input_size = topic_embed_size + 4 + 4 + ctx_hidden_size
    recog_input_size = prior_input_size + utt_hidden_size * 2
    if use_hcf:
        recog_input_size += 30
    latent_size = 200

    gen_inputs_size = prior_input_size + latent_size

    dec_num_layer = 1
    dec_inputs_size = gen_inputs_size
    if use_hcf:
        dec_inputs_size += 30
    dec_hidden_size = 400

    dec_input_embedding_size = word_embed_size
    if use_hcf:
        dec_input_embedding_size += 30

    step_size = 1                                  # internal usage
    batch_size = 60                                # mini-batch size
    backward_size = 10                             # how many utterance kept in the context window

    init_lr = 0.001                                # initial learning rate
    keep_prob = 1.0                                # drop out rate
    dec_keep_prob = 1.0