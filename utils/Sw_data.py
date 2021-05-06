

import os
import numpy as np
import pickle as pkl
from utils.alphabet import Alphabet
from utils.functions import normalizeStringNLTK


class Data:
    def __init__(self, corpus_file):
        self.data_dict = pkl.load(open(corpus_file, "rb"))
        self.train_corpus = self.process(self.data_dict["train"])
        self.valid_corpus = self.process(self.data_dict["valid"])
        self.test_corpus = self.process(self.data_dict["test"])

        self.word_alphabet = Alphabet('word')
        self.topic_alphabet = Alphabet('topic')
        self.act_alphabet = Alphabet('act')
        self.build_alphabet()

        self.utt_corpus = self.get_utt_corpus()
        self.meta_corpus = self.get_meta_corpus()
        self.dialog_corpus = self.get_dialog_corpus()

        print("Done loading corpus!")

    def process(self, data):
        new_dialog = []
        new_meta = []
        new_utts = []
        all_lengths = []

        for line in data:
            lower_utts = [(caller, normalizeStringNLTK(utt).split(" "), feat)
                          for caller, utt, feat in line["utts"]]
            all_lengths.extend([len(u) for c, u, f in lower_utts])

            a_age = float(line["A"]["age"]) / 100.0
            b_age = float(line["B"]["age"]) / 100.0
            a_edu = float(line["A"]["education"]) / 3.0
            b_edu = float(line["B"]["education"]) / 3.0
            vec_a_meta = [a_age, a_edu] + ([0, 1] if line["A"]["sex"] == "FEMALE" else [1, 0])
            vec_b_meta = [b_age, b_edu] + ([0, 1] if line["B"]["sex"] == "FEMALE" else [1, 0])

            # eg. ([0.5, 0.6666666666666666, 1, 0], [0.75, 0.6666666666666666, 1, 0], '"BUYING A CAR"')
            meta = (vec_a_meta, vec_b_meta, line["topic"])
            # 1 is own utt and 0 is other's utt
            dialog = [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]

            new_utts.extend([utt for caller, utt, feat in lower_utts])
            new_meta.append(meta)
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lengths), float(np.mean(all_lengths))))
        return new_dialog, new_meta, new_utts

    def build_alphabet(self):
        # create word alphabet
        for tokens in self.train_corpus[2]:
            self.word_alphabet.addTokenList(tokens)
        print("%d words in train data" % self.word_alphabet.num_tokens)

        # create topic alphabet
        for a, b, topic in self.train_corpus[1]:
            self.topic_alphabet.addToken(topic)
        print("%d topics in train data" % self.topic_alphabet.num_tokens)

        # get dialog act labels
        for dialog in self.train_corpus[0]:
            for caller, utt, feat in dialog:
                if feat is not None: 
                    self.act_alphabet.addToken(feat[0])
        print("%d dialog acts in train data" % self.act_alphabet.num_tokens)

    def get_utt_corpus(self):
        """convert utterance into numbers
        """
        id_train = [self.word_alphabet.list_to_indexes(sentence) for sentence in self.train_corpus[2]]
        id_valid = [self.word_alphabet.list_to_indexes(sentence) for sentence in self.valid_corpus[2]]
        id_test = [self.word_alphabet.list_to_indexes(sentence) for sentence in self.test_corpus[2]]
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_meta_corpus(self):
        """convert topic into numbers
        """
        id_train = [(a, b, self.topic_alphabet.get_index(topic)) for a, b, topic in self.train_corpus[1]]
        id_valid = [(a, b, self.topic_alphabet.get_index(topic)) for a, b, topic in self.valid_corpus[1]]
        id_test = [(a, b, self.topic_alphabet.get_index(topic)) for a, b, topic in self.test_corpus[1]]
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        """convert utterance and feature into numeric numbers
        """
        def _to_id_corpus(corpus):
            results = []
            for dialog in corpus:
                temp = []
                for utt, floor, feat in dialog:
                    id_feat = None
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[0] = self.act_alphabet.get_index(feat[0])
                    temp.append((self.word_alphabet.list_to_indexes(utt), floor, id_feat))
                results.append(temp)
            return results
        id_train = _to_id_corpus(self.train_corpus[0])
        id_valid = _to_id_corpus(self.valid_corpus[0])
        id_test = _to_id_corpus(self.test_corpus[0])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def data_generator(self, batch_size):
        pass


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")

    obj = Data(corpus_file)
