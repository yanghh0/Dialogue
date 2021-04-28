
import math
import numpy as np


NEG_INF = -float("inf")


def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def forward_log(y, labels):
    log_y = np.log(y)

    T, V = log_y.shape
    L = len(labels)
    log_alpha = np.ones([T, L]) * NEG_INF

    # init
    log_alpha[0, 0] = log_y[0, labels[0]]
    log_alpha[0, 1] = log_y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            temp = log_alpha[t - 1, i]
            if i - 1 >= 0:
                temp = logsumexp(temp, log_alpha[t - 1, i - 1])
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                temp = logsumexp(temp, log_alpha[t - 1, i - 2])
            log_alpha[t, i] = temp + log_y[t, s]

    return log_alpha


def backward_log(y, labels):
    log_y = np.log(y)

    T, V = log_y.shape
    L = len(labels)
    log_beta = np.ones([T, L]) * NEG_INF

    # init
    log_beta[-1, -1] = log_y[-1, labels[-1]]
    log_beta[-1, -2] = log_y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]

            temp = log_beta[t + 1, i]
            if i + 1 < L:
                temp = logsumexp(temp, log_beta[t + 1, i + 1])
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                temp = logsumexp(temp, log_beta[t + 1, i + 2])
            log_beta[t, i] = temp + log_y[t, s]

    return log_beta
    

if __name__ == "__main__":
    np.random.seed(3)
    y = softmax(np.random.random([10, 5]))
    # print(y)
    # print(y.sum(axis=1, keepdims=True))

    labels = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank

    log_alpha = forward_log(y, labels)
    log_beta = backward_log(y, labels)
    # print(log_alpha)
    # print(log_beta)

    log_p_alpha = logsumexp(log_alpha[-1, labels[-1]], log_alpha[-1, labels[-2]])
    log_p_beta  = logsumexp(log_beta[0, labels[0]], log_beta[0, labels[1]])
    print(log_p_alpha)
    print(log_p_beta)