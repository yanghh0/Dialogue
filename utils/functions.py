
import re
import unicodedata
import itertools


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


if __name__ == "__main__":
    s = " do you like china? it's a good place.\r\n"
    s = normalizeString(s)
    print(s)

    inputs = [[1, 2, 3, 4],
              [4, 3, 2],
              [7, 4]]
    outputs = zeroPadding(inputs)
    print(outputs)

    m = binaryMatrix(outputs)
    print(m)
