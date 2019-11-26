import os
import random
import numpy as np
from PIL import Image
from data.generic_dataset import GenericDataset


class CycleGanDataset(GenericDataset):

    def __init__(self, root, *args, random_pairs=True, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        """
        *args: data_paths (str, list or tuple): full path/paths to root dir/dirs from where
                          the local file paths must be collected
               extensions (str or tuple): format of images to be collected ('.jpeg', '.jpg', '.png', etc.)
        root (str): root directory with the resource files
        transform (callable, optional): Optional transform to be applied
                          on a sample.
        """
        self.root_dir = root
        self.random_pairs = random_pairs
        self.transform = transform

        # we assume that self._found_dataset contains two dicts: first one for source images (trainA or testA)
        # second one for target images (trainB or testB)

        source_imgs = [os.path.join(os.path.basename(self._found_dataset[0]['root']), name)
                       for name in self._found_dataset[0]['names']]
        target_imgs = [os.path.join(os.path.basename(self._found_dataset[1]['root']), name)
                       for name in self._found_dataset[1]['names']]

        # slice both datasets based on the minimum length
        if len(source_imgs) != len(target_imgs):
            min_len = min(len(source_imgs), len(target_imgs))
            source_imgs = source_imgs[:min_len]
            target_imgs = target_imgs[:min_len]

        self.source_imgs = source_imgs
        self.target_imgs = target_imgs

    def __len__(self):
        return len(self.source_imgs)

    def __getitem__(self, item):
        """
           Returns unpaired source and target images
        """
        img_name_source = os.path.join(self.root_dir, self.source_imgs[item])

        if self.random_pairs:
            # second image for pair is selected randomly
            random_b = random.randint(0, len(self.source_imgs)-1)
            img_name_target = os.path.join(self.root_dir, self.target_imgs[random_b])
        else:
            img_name_target = os.path.join(self.root_dir, self.target_imgs[item])

        img_source = np.asarray(Image.open(img_name_source))
        img_target = np.asarray(Image.open(img_name_target))

        # to deal with grayscale images (1 channel instead of 3)
        if len(img_source.shape) != 3:
            img_source = np.repeat(img_source[:, :, np.newaxis], 3, axis=2)
        if len(img_target.shape) != 3:
            img_target = np.repeat(img_target[:, :, np.newaxis], 3, axis=2)

        # apply transforms
        if self.transform:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target
