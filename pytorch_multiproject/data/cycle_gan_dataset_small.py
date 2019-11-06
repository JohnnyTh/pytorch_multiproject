import numpy as np
import random


class CycleGanDatasetSmall:

    def __init__(self, dataset_one, dataset_two, sample_size, transform=None):
        """
           Description here
        """
        self.transform = transform
        self.dataset_one = dataset_one
        self.dataset_two = dataset_two
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __getitem__(self, item):
        """
           Returns unpaired source and target images
        """

        img_source = np.asarray(self.dataset_one[item][0])

        random_b = random.randint(0, len(self.sample_size)-1)
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
