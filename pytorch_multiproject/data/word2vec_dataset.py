import os
import torch
import pickle
import random
# import numpy as np
from data.generic_dataset import GenericDataset

# NOTE - quoted lines of code use numpy instead of for loop for subsampling


class Word2VecDataset(GenericDataset):

    def __init__(self, root, word2idx, idx2word, word_freq=None, subsamp_thresh=10**(-5), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_addr = os.path.join(self._found_dataset[0]['root'], self._found_dataset[0]['names'][0])
        self.word_freq = word_freq
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.subsampl_thresh = torch.tensor(subsamp_thresh).float()
        self.subsampl_prob = None
        if self.word_freq is not None:
            subsampl_prob = 1 - torch.sqrt(self.subsampl_thresh / self.word_freq)
            # two first elements in subsampling prob distriution have value >1 so that they are never selected
            # since 0 - padding el, 1 - unk el
            self.subsampl_prob = torch.cat([torch.tensor([1.1, 1.1]),
                                            torch.clamp(subsampl_prob, 0, 1)])
            # self.subsampl_prob = torch.cat([torch.tensor([1.1, 1.1]),
            #                                 torch.clamp(subsampl_prob, 0, 1)]).numpy()
        self.data = None

    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, item):
        input_word, target_words = self.data[item]
        return torch.tensor(input_word).long(), torch.tensor(target_words).long()

    def subsample_or_get_data(self):
        data = pickle.load(open(self.data_addr, 'rb'))
        # data = np.array(pickle.load(open(self.data_addr, 'rb'))
        # if word frequency is supplied, drop input words with certain probability
        data_balanced = []
        if self.word_freq is not None:
            for input_word, target_words in data:
                if random.random() > self.subsampl_prob[input_word]:
                    data_balanced.append((input_word, target_words))
            data = data_balanced

        # use subsampl_prob as lookup table for input words in data
        # all_probs = self.subsampl_prob[data[:, 0]]
        # threshold = np.random.rand(len(all_probs))
        # mask = threshold > all_probs
        # self.data = data[mask]
        self.data = data
