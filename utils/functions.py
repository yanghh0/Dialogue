
import os
import re
import glob
import torch
import unicodedata
import itertools
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def get_bleu_stats(ref, hyps):
    scores = []
    for hyp in hyps:
        try:
            scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3,1./3]))
        except:
            scores.append(0.0)
    return np.max(scores), np.mean(scores)


def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    # 在标点前加空格，这样标点符号就变成一个token.
    s = re.sub(r"([.!?])", r" \1", s)
    # 字母和标点外的字符都变成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # 把多个连续空格变成一个空格，并去掉首尾空格
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalizeStringNLTK(s):
    s = nltk.WordPunctTokenizer().tokenize(s.lower())
    return " ".join(s)


def zeroPadding(inputs, fillvalue=0):
    return list(itertools.zip_longest(*inputs, fillvalue=fillvalue))


def binaryMatrix(inputs, pad_token=0):
    m = []
    for i, seq in enumerate(inputs):
        m.append([])
        for token in seq:
            if token == pad_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def pad_to(sequence, max_length, do_pad=True):
    if len(sequence) >= max_length:
        return sequence[0:max_length-1] + [sequence[-1]]
    elif do_pad:
        return sequence + [0] * (max_length - len(sequence))
    return sequence


def print_model_stats(tvars):
    total_parameters = 0
    for name, param in tvars:
        shape = param.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        print("Trainable %s with %d parameters" % (name, variable_parameters))
        total_parameters += variable_parameters
    print("Total number of trainable parameters is %d" % total_parameters)


def sample_gaussian(mu, logvar):
    epsilon = logvar.new_empty(logvar.size()).normal_()
    std = torch.exp(0.5 * logvar)
    z = mu + std * epsilon
    return z


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 - (prior_logvar - recog_logvar)
                             - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                             - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


def print_loss(prefix, loss_names, losses, postfix):
    template = "%s "
    for name in loss_names:
        template += "%s " % name
        template += " %f "
    template += "%s"
    template = re.sub(' +', ' ', template)
    avg_losses = []
    values = [prefix]

    for loss in losses:
        values.append(np.mean(loss))
        avg_losses.append(np.mean(loss))
    values.append(postfix)

    print(template % tuple(values))
    return avg_losses


def get_checkpoint_state(ckp_dir):
    files = os.path.join(ckp_dir, "*.pth")
    files = glob.glob(files)
    files.sort(key=os.path.getmtime)
    return len(files) > 0 and files[-1] or None


if __name__ == "__main__":
    s = " do you like china? it's a good place. EOS\r\n"
    print(normalizeString(s))
    print(normalizeStringNLTK(s))
    print(pad_to(normalizeStringNLTK(s).split(" "), 5))
    print(pad_to(normalizeStringNLTK(s).split(" "), 20))

    inputs = [[1, 2, 3, 4],
              [4, 3, 2],
              [7, 4]]
    outputs = zeroPadding(inputs)
    print(outputs)

    m = binaryMatrix(outputs)
    print(m)


