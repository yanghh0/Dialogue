
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import numpy as np
import pickle as pkl
from seq2seq_cvae.config import Config
from utils.alphabet import Alphabet
from utils.functions import normalizeStringNLTK, pad_to


class Data:
    def __init__(self, corpus_file, mode):
        self.data_dict = pkl.load(open(corpus_file, "rb"))
        self.train_corpus = self.process(self.data_dict["train"])
        self.valid_corpus = self.process(self.data_dict["valid"])
        self.test_corpus = self.process(self.data_dict["test"])

        self.word_alphabet = Alphabet('word')
        self.topic_alphabet = Alphabet('topic')
        self.act_alphabet = Alphabet('act')
        self.build_alphabet()
        self.load_word2vec()

        self.utt_corpus_dict = self.get_utt_corpus()
        self.meta_corpus_dict = self.get_meta_corpus()
        self.dialog_corpus_dict = self.get_dialog_corpus()

        self.s_metas = self.meta_corpus_dict[mode]
        self.s_dialogs = self.dialog_corpus_dict[mode]
        self.s_lengths = [len(dialog) for dialog in self.s_dialogs]
        self.s_indexes = list(np.argsort(self.s_lengths))

        self.pointer = 0
        self.num_batch = 0
        self.epoch_data = []
        self.grid_indexes = []
        self.batch_size = 1 if mode == "test" else Config.batch_size

        print("Done loading corpus!")

    def process(self, calls):
        calls_dialogs = []
        calls_metas = []
        calls_utts = []
        all_lengths = []

        for one_call in calls:
            lower_utts = [(caller, normalizeStringNLTK(utt).split(" "), feat)
                          for caller, utt, feat in one_call["utts"]]
            all_lengths.extend([len(u) for c, u, f in lower_utts])

            a_age = float(one_call["A"]["age"]) / 100.0
            b_age = float(one_call["B"]["age"]) / 100.0
            a_edu = float(one_call["A"]["education"]) / 3.0
            b_edu = float(one_call["B"]["education"]) / 3.0
            vec_a_meta = [a_age, a_edu] + ([0, 1] if one_call["A"]["sex"] == "FEMALE" else [1, 0])
            vec_b_meta = [b_age, b_edu] + ([0, 1] if one_call["B"]["sex"] == "FEMALE" else [1, 0])

            # eg. ([0.5, 0.6666666666666666, 1, 0], [0.75, 0.6666666666666666, 1, 0], '"BUYING A CAR"')
            meta = (vec_a_meta, vec_b_meta, one_call["topic"])
            # 1 is own utt and 0 is other's utt
            dialog = [(utt, int(caller=="B"), feat) for caller, utt, feat in lower_utts]

            calls_utts.extend([utt for caller, utt, feat in lower_utts])
            calls_metas.append(meta)
            calls_dialogs.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lengths), float(np.mean(all_lengths))))
        return calls_dialogs, calls_metas, calls_utts

    def build_alphabet(self):
        # create word alphabet
        for tokens in self.train_corpus[2]:
            self.word_alphabet.addTokenList(tokens)
        Config.word_vocab_size = self.word_alphabet.num_tokens
        print("%d words in train data" % self.word_alphabet.num_tokens)

        # create topic alphabet
        for a, b, topic in self.train_corpus[1]:
            self.topic_alphabet.addToken(topic)
        Config.topic_vocab_size = self.topic_alphabet.num_tokens
        print("%d topics in train data" % self.topic_alphabet.num_tokens)

        # get dialog act labels
        for dialog in self.train_corpus[0]:
            for caller, utt, feat in dialog:
                if feat is not None: 
                    self.act_alphabet.addToken(feat[0])
        Config.act_vocab_size = self.act_alphabet.num_tokens
        print("%d dialog acts in train data" % self.act_alphabet.num_tokens)

    def load_word2vec(self):
        if Config.word_vec_path is None:
            return
        with open(Config.word_vec_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.decode().split(" ", 1)
            raw_word2vec[w] = vec
        # clean up lines for memory efficiency
        word2vec = []
        oov_cnt = 0
        for v in self.word_alphabet.index2token:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(Config.word_embed_size) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            word2vec.append(vec)
        Config.word2vec = word2vec
        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.word_alphabet.index2token)))

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

    def epoch_init(self, shuffle=True, intra_shuffle=True):
        self.pointer = 0

        temp_num_batch = len(self.s_dialogs) // self.batch_size
        self.epoch_data = []

        for i in range(temp_num_batch):
            self.epoch_data.append(self.s_indexes[i*self.batch_size : (i+1)*self.batch_size])
        if shuffle:
            np.random.shuffle(self.epoch_data)

        self.grid_indexes = []
        for idx, b_ids in enumerate(self.epoch_data):
            b_lengths = [self.s_lengths[rank] for rank in b_ids]   # 批次内所有通话时长
            max_b_length = self.s_lengths[b_ids[-1]]               # 批次内最大通话时长
            min_b_length = self.s_lengths[b_ids[0]]                # 批次内最短通话时长

            num_seg = (max_b_length - Config.backward_size) // Config.step_size

            if num_seg > 0:
                cut_start = list(range(0, num_seg*Config.step_size, Config.step_size))
                cut_end = list(range(Config.backward_size, num_seg*Config.step_size+Config.backward_size, Config.step_size))
                cut_start = [0] * (Config.backward_size-2) + cut_start
                cut_end = list(range(2, Config.backward_size)) + cut_end
            else:
                cut_start = [0] * (max_b_length - 2)
                cut_end = list(range(2, max_b_length))

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_b_length-1]
            if intra_shuffle and shuffle:
                np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)
        self.num_batch = len(self.grid_indexes)

    def next_batch(self):
        if self.pointer < self.num_batch:
            current_grid = self.grid_indexes[self.pointer]
            self.pointer += 1
            return self._prepare_batch(cur_grid=current_grid)
        else:
            return None

    def _prepare_batch(self, cur_grid):
        b_id, s_id, e_id = cur_grid

        batch_data = self.epoch_data[b_id]
        batch_dialog = [self.s_dialogs[idx] for idx in batch_data]
        batch_meta = [self.s_metas[idx] for idx in batch_data]
        batch_topic = np.array([meta[2] for meta in batch_meta])

        assert self.batch_size == len(batch_data)

        context_utts = []
        floors = []
        context_lens = []
        out_utts = []
        out_lens = []
        out_floors = []
        out_des = []

        for dialog in batch_dialog:
            if s_id < len(dialog) - 1:
                inputs_output = dialog[s_id:e_id]
                inputs = inputs_output[0:-1]
                output = inputs_output[-1]
                out_utt, out_floor, out_feat = output

                """
                context_utts[i]: [[], [], ...]
                floors[i]: [0, 1, 1, ]
                context_lens[i]: scalar < 10 or 10
                """
                context_utts.append([pad_to(utt, Config.max_utt_length) for utt, floor, feat in inputs])
                floors.append([int(floor == out_floor) for utt, floor, feat in inputs])
                context_lens.append(len(inputs_output) - 1)

                """
                out_utt: []
                out_utts[i]: []
                out_lens[i]: scalar
                out_floors[i]: 0 or 1
                out_des[i]: scalar
                """
                out_utt = pad_to(out_utt, Config.max_utt_length, do_pad=False)
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))
                out_floors.append(out_floor)
                out_des.append(out_feat[0])
            else:
                print(dialog)
                raise ValueError("S_ID %d larger than dialog" % s_id)

        """
        [[0.73       0.66666667 1.         0.        ]
         [0.59       0.66666667 1.         0.        ]
         ...
        """
        my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(batch_meta)])
        ot_profiles = np.array([meta[1-out_floors[idx]] for idx, meta in enumerate(batch_meta)])

        context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(context_lens), Config.max_utt_length), dtype=np.int32)
        vec_floors = np.zeros((self.batch_size, np.max(context_lens)), dtype=np.int32)

        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_out_des = np.array(out_des)

        for b_id in range(self.batch_size):
            vec_context[b_id, 0:context_lens[b_id], :] = np.array(context_utts[b_id])
            vec_floors[b_id, 0:context_lens[b_id]] = floors[b_id]
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]

        return torch.from_numpy(vec_context).long(),     \
               torch.from_numpy(context_lens).long(),    \
               torch.from_numpy(vec_floors).long(),      \
               torch.from_numpy(batch_topic).long(),     \
               torch.from_numpy(my_profiles).float(),    \
               torch.from_numpy(ot_profiles).float(),    \
               torch.from_numpy(vec_outs).long(),        \
               torch.from_numpy(vec_out_lens).long(),    \
               torch.from_numpy(vec_out_des).long()


if __name__ == "__main__":
    corpus_name = "Switchboard(SW) 1 Release 2 Corpus"
    corpus_file = os.path.join("..", "datasets", corpus_name, "full_swda_clean_42da_sentiment_dialog_corpus.p")

    obj = Data(corpus_file, "train")
    obj.epoch_init()
    iteration = 0
    while True:
        batch_data = obj.next_batch()
        if batch_data is None:
            break
        iteration += 1

