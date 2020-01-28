import os
import torch
import pickle
import random
import numpy as np
from data.generic_dataset import GenericDataset


class Word2VecDataset(GenericDataset):

    def __init__(self, word_freq=None, subsamp_thresh=10**(-5), subsamp_implementation='loop',
                 *args, **kwargs):
        """
        Returns pairs input word - context words from prepared dataset.
        Parameters
        ----------
        word_freq (torch.tensor): a tensor of length == vocab length, contains word frequencies for each word in vocab.
        subsamp_thresh (float): a threshold value; words above this frequency will be actively suppressed
        subsamp_implementation (str, "loop" or "numpy"): selection between implementation of subsampling using for loop
            (slower, consumes less memory) or numpy array (faster, consumes more memory)
        args, kwargs: : data_paths, extensions
        """
        super().__init__(*args, **kwargs)
        self.data_addr = os.path.join(self._found_dataset[0]['root'], self._found_dataset[0]['names'][0])
        self.word_freq = word_freq
        self.subsampl_thresh = torch.tensor(subsamp_thresh).float()
        self.subsampl_prob = None
        self.subsamp_implementation = subsamp_implementation
        if self.word_freq is not None:
            subsampl_prob = 1 - torch.sqrt(self.subsampl_thresh / self.word_freq)
            # two first elements in subsampling prob distriution have value >1 so that they are never selected
            # since 0 - padding el, 1 - unk el
            if self.subsamp_implementation == 'loop':
                self.subsampl_prob = torch.cat([torch.tensor([1.1, 1.1]),
                                                torch.clamp(subsampl_prob, 0, 1)])
            elif self.subsamp_implementation == 'numpy':
                self.subsampl_prob = torch.cat([torch.tensor([1.1, 1.1]),
                                                torch.clamp(subsampl_prob, 0, 1)]).numpy()
            else:
                raise ValueError('subsamp_implementation can be either "loop" or "numpy", but {} was provided!'
                                 .format(self.subsamp_implementation))
        self.data = None

    def __len__(self):
        length = len(self.data)
        return length

    def __getitem__(self, item):
        input_word, target_words = self.data[item]
        return torch.tensor(input_word).long(), torch.tensor(target_words).long()

    def subsample_or_get_data(self):
        # upon call resamples the dataset, discarding input - target pairs where input is a frequently encountered
        # word with high probability
        data = None

        if self.subsamp_implementation == 'loop':
            data = pickle.load(open(self.data_addr, 'rb'))
            # if word frequency is supplied, drop input words with certain probability
            data_balanced = []
            if self.word_freq is not None:
                for input_word, target_words in data:
                    if random.random() > self.subsampl_prob[input_word]:
                        data_balanced.append((input_word, target_words))
                data = data_balanced

        elif self.subsamp_implementation == 'numpy':
            data = np.array(pickle.load(open(self.data_addr, 'rb')))
            if self.word_freq is not None:
                # use subsampl_prob as lookup table for input words in data
                all_probs = self.subsampl_prob[data[:, 0]]
                threshold = np.random.rand(len(all_probs))
                mask = threshold > all_probs
                # select only values are below the threshold
                data = data[mask]

        self.data = data
