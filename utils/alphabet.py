
# Default tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token


class Alphabet:
    """Alphabet maps objects to integer ids.
    """
    def __init__(self, name):
        self.name = name         # word/char/topic 
        self.trimmed = False
        self.initialize()

    def addTokenSeqence(self, token_sequence):
        for token in token_sequence.split(' '):
            self.addToken(token)

    def addTokenList(self, token_list):
        for token in token_list:
            self.addToken(token)

    def addToken(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.num_tokens
            self.index2token.append(token)
            self.token2count[token] = 1
            self.num_tokens += 1
        else:
            self.token2count[token] += 1

    def initialize(self):
        self.token2index = {}
        self.token2count = {}
        self.index2token = []
        self.num_tokens = 0
        if self.name == 'word':
            self.index2token = ["PAD", "SOS", "EOS", "UNK"]
            self.num_tokens = 4

    def get_content(self):
        return {'token2index': self.token2index, 
                'index2token': self.index2token,
                'token2count': self.token2count}

    def get_index(self, token):
        try:
            return self.token2index[token]
        except KeyError:
            return self.token2index["UNK"]

    def sequence_to_indexes(self, token_seqence):
        return [self.get_index(token) for token in token_seqence.split(' ')] + [EOS_token]

    def top_n_token(self, n=-1):
        my_list = list(self.token2count.items())
        my_list.sort(key=lambda x: x[1], reverse=True)
        return my_list[:n]

    def trim(self, min_count):
        """Remove tokens below a certain count threshold
        """
        if self.trimmed:
            return
        self.trimmed = True
        keep_tokens = []
        for k, v in self.token2count.items():
            if v >= min_count:
                keep_tokens.append(k)

        print('keep_tokens {} / {} = {:.4f}'.format(
            len(keep_tokens), len(self.token2count), len(keep_tokens) / len(self.token2count)
        ))

        self.initialize()
        for token in keep_tokens:
            self.addToken(token)


if __name__ == "__main__":
    word_alphabet = Alphabet("word")
    word_alphabet.addTokenSeqence("how old are you")
    word_alphabet.addTokenSeqence("do you like China")
    word_alphabet.addTokenSeqence("how about China")

    context_dict = word_alphabet.get_content()
    print(context_dict)

    print(word_alphabet.sequence_to_indexes("how old are you"))

    word_alphabet.trim(min_count=2)
    context_dict = word_alphabet.get_content()
    print(context_dict)
