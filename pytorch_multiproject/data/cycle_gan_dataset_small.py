import os
import logging
import random
import numpy as np


class CycleGanDatasetSmall:

    def __init__(self, dataset_one, dataset_two, sample_size, transform=None):
        """Dataset implementation for MNIST to SVHN GAN
           dataset_one, dataset_two (sequence): datasets with source and target images.
           sample_size (int): number of samples we want to use from both datasets.
           transform (callable, optional): Optional transform to be applied on an image.
        """
        self.dataset_one = dataset_one
        self.dataset_two = dataset_two
        self.sample_size = sample_size
        self.transform = transform
        self.random_indices = self._get_random_indices()
        self.logger = logging.getLogger(os.path.basename(__file__))

    def __len__(self):
        # len methods is required for dataloader and defines when for loop stops
        return self.sample_size

    def __getitem__(self, item):
        """
           Returns unpaired source and target images
        """
        img_source = np.asarray(self.dataset_one[item][0])

        # pop random indices from list until it's depleted
        try:
            random_b = self.random_indices.pop()
        # if list is depleted, one epoch has passed - generate a new list of indices
        except IndexError:
            self.logger.debug('Random indices depleted, generating a new batch')
            self.random_indices = self._get_random_indices()
            random_b = self.random_indices.pop()

        img_target = np.asarray(self.dataset_two[random_b][0])

        # to deal with grayscale images (1 channel instead of 3)
        if len(img_source.shape) != 3:
            img_source = np.repeat(img_source[:, :, np.newaxis], 3, axis=2)
        if len(img_target.shape) != 3:
            img_target = np.repeat(img_target[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def _get_random_indices(self):
        # generates a list of random indices for selection of random image pairs
        indices = [i for i in range(self.sample_size)]
        random.shuffle(indices)
        return indices
