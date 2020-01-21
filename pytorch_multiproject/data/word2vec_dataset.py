import os
import torch
import pickle
import random
import numpy as np
from data.generic_dataset import GenericDataset


class Word2VecDataset(GenericDataset):

    def __init__(self, root, word2idx, idx2word, word_freq=None, subsamp_thresh=10**(-5), *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.data_addr = os.path.join(root, self._found_dataset[0]['names'][0])
        self.data = self.get_data()
        self.word2idx = word2idx
        self.idx2word = idx2word

        self.word_freq = word_freq
        self.subsampl_thresh = subsamp_thresh
        subsampl_prob = 1 - np.sqrt(self.subsampl_thresh / self.word_freq)
        self.subsampl_prob = np.clip(subsampl_prob, 0, 1)

        self.memory = {'last_idx': None, 'current_idx': None}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.memory['last_idx'] is None:
            self.memory['last_idx'] = item
        self.memory['current_idx'] = item
        # resample the data after each epoch
        if (self.memory['current_idx'] == 0
            and self.memory['last_idx'] == len(self.data) - 1
            and self.word_freq is not None):
            self.data = self.get_data()

        input_word, target_words = self.data[item]
        self.memory['last_idx'] = self.memory['current_idx']
        return input_word, np.array(target_words)

    def get_data(self):
        data = pickle.load(self.data_addr)
        # if word frequency is supplied, drop input words with certain probability
        if self.word_freq is not None:
            data_balanced = []
            for input_word, target_words in data:
                if random.random() > self.subsampl_prob[input_word]:
                    data_balanced.append((input_word, target_words ))
            data = data_balanced

        return data
