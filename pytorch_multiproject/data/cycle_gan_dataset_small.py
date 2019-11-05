import numpy as np


class CycleGanDatasetSmall:

    def __init__(self, mnist, svhn, sample_size, transform=None):
        """
           Description here
        """
        self.transform = transform
        self.mnist = mnist
        self.svhn = svhn
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __getitem__(self, item):
        """
           Returns unpaired source and target images
        """

        img_source = np.asarray(self.mnist[item][0])
        img_target = np.asarray(self.svhn[item][0])

        # to deal with grayscale images (1 channel instead of 3)
        if len(img_source.shape) != 3:
            img_source = np.repeat(img_source[:, :, np.newaxis], 3, axis=2)
        if len(img_target.shape) != 3:
            img_target = np.repeat(img_target[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target
