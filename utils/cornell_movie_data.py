
import sys
sys.path.append(".")
sys.path.append("..")


import os
import torch
import random
import numpy as np
from tqdm import tqdm
from utils.alphabet import Alphabet
from seq2seq_gru.config import Config
from utils.functions import normalizeString, zeroPadding, binaryMatrix


class Data:
    def __init__(self):
        self.word_alphabet = Alphabet('word')
        self.pairs = []

        self.pointer = 0
        self.num_batch = 0
        self.epoch_data = []

    def filterPair(self, pair):
        return len(pair[0].split(' ')) < Config.max_sentence_length and len(pair[1].split(' ')) < Config.max_sentence_length

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def build_alphabet(self, datafile):
        print("Start preparing training data ...")
        lines = open(datafile, mode='r', encoding='utf-8').readlines()

        for l in tqdm(lines):
            self.pairs.append([normalizeString(s) for s in l.strip().split('\t')])
        print("Read {!s} sentence pairs".format(len(self.pairs)))

        self.pairs = self.filterPairs(self.pairs)
        print("Trimmed to {!s} sentence pairs".format(len(self.pairs)))

        print("Build alphabet...")
        for pair in self.pairs:
            self.word_alphabet.addTokenSeqence(pair[0])
            self.word_alphabet.addTokenSeqence(pair[1])
        print("Counted words:", self.word_alphabet.num_tokens)

    def trimRareWords(self, min_count=3):
        """ 1. 使用 word_alphabet.trim() 函数去掉频次低于 min_count 的词。
            2. 去掉包含低频词的句子，即保留的句子的每一个词都必须是高频的。
        """
        # Trim words used under the min_count from the word_alphabet
        self.word_alphabet.trim(min_count)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in self.pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True

            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in self.word_alphabet.token2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in self.word_alphabet.token2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(self.pairs), len(keep_pairs), 
              len(keep_pairs) / len(self.pairs)))

        self.pairs = keep_pairs

    def process_inputs(self, inputs):
        """Returns padded input sequence tensor and lengths
        """
        indexes_inputs = [self.word_alphabet.sequence_to_indexes(sentence) for sentence in inputs]
        inputs_length = torch.LongTensor([len(indexes) for indexes in indexes_inputs])
        paded_inputs = zeroPadding(indexes_inputs)
        paded_inputs = torch.LongTensor(paded_inputs)
        return paded_inputs, inputs_length

    def process_targets(self, targets):
        """Returns padded target sequence tensor, padding mask, and max target length
        """
        indexes_targets = [self.word_alphabet.sequence_to_indexes(sentence) for sentence in targets]
        max_target_len = max([len(indexes) for indexes in indexes_targets])
        paded_targets = zeroPadding(indexes_targets)
        mask = binaryMatrix(paded_targets)
        mask = torch.ByteTensor(mask).bool()
        paded_targets = torch.LongTensor(paded_targets)
        return paded_targets, mask, max_target_len

    def batch2TrainData(self, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        inputs, targets = [], []
        for pair in pair_batch:
            inputs.append(pair[0])
            targets.append(pair[1])
        inputs, inputs_lengths = self.process_inputs(inputs)
        targets, mask, max_target_len = self.process_targets(targets)
        return inputs, inputs_lengths, targets, mask, max_target_len

    def epoch_init(self, batch_size):
        self.num_batch = len(self.pairs) // batch_size
        self.pointer = 0
        self.epoch_data = []

        for i in range(self.num_batch):
            self.epoch_data.append(self.pairs[i*batch_size : (i+1)*batch_size])
        np.random.shuffle(self.epoch_data)

    def next_batch(self):
        if self.pointer < self.num_batch:
            batch_data = self._prepare_batch()
            self.pointer += 1
            return batch_data
        else:
            return None

    def _prepare_batch(self):
        return self.batch2TrainData(self.epoch_data[self.pointer])


if __name__ == "__main__":
    corpus_name = "cornell movie-dialogs corpus"
    datafile = os.path.join("..", "datasets", corpus_name, "formatted_movie_lines.txt")

    obj = Data()
    obj.build_alphabet(datafile)

    # print("\npairs:")
    # for pair in obj.pairs[:10]:
    #     print(pair)

    obj.trimRareWords()

    small_batch_size = 3
    ori_inputs = [random.choice(obj.pairs) for _ in range(small_batch_size)]
    batches = obj.batch2TrainData(ori_inputs)
    inputs, inputs_lengths, targets, mask, max_target_len = batches

    print("ori_inputs:", ori_inputs)
    print("inputs:", inputs)
    print("inputs_lengths:", inputs_lengths)
    print("targets:", targets)
    print("mask:", mask)
    print("max_target_len:", max_target_len)

    obj.epoch_init(batch_size=128)
    while True:
        if obj.next_batch() is None:
            break
