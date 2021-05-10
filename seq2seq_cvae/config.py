
import os
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
    word2vec = []

    word_embed_size = 200
    topic_embed_size = 30
    act_embed_size = 30

    word_vec_path = os.path.join("..", "glove", "glove.6B.200d.txt")

    utt_num_layer = 1
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
        dec_inputs_size += act_embed_size
    dec_hidden_size = 400
    dec_input_embed_size = word_embed_size
    if use_hcf:
        dec_input_embed_size += act_embed_size

    step_size = 1                # internal usage
    batch_size = 60              # mini-batch size
    backward_size = 10           # how many utterance kept in the context window

    init_lr = 0.001              # initial learning rate
    keep_prob = 1.0              # drop out rate
    dec_keep_prob = 1.0
    full_kl_step = 10000         # how many batch before KL cost weight reaches 1.0
    max_epoch = 60               # max number of epoch of training
    grad_clip = 5.0              # gradient abs max cut
    init_w = 0.08                # uniform random from [-init_w, init_w]

    improve_threshold = 0.996    # for early stopping
    patient_increase = 2.0       # for early stopping
    early_stop = True