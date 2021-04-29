
import numpy as np


def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def beam_decode(y, beam_size=3):
    T, V = y.shape
    log_y = np.log(y)  # 对数概率

    beam = [([], 0)]
    for t in range(T): # 遍历每个时间步
        new_beam = []  # 存储增加新字母后的候选序列
        for prefix, p in beam:  # 遍历已经生成的序列
            for i in range(V):  # 遍历字母表内每个字母
                new_prefix = prefix + [i]
                new_p = p + log_y[t, i]

                new_beam.append((new_prefix, new_p))

        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    return beam


if __name__ == "__main__":
    np.random.seed(3)
    y = softmax(np.random.random([20, 6]))
    beam = beam_decode(y, beam_size=100)
    for string, p in beam[:20]:
        print(remove_blank(string), p)