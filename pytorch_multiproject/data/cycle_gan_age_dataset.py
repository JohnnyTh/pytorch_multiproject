import os
import random
import numpy as np
from PIL import Image
from data.generic_dataset import GenericDataset


class AgeGanDataset(GenericDataset):

    def __init__(self, full_df, root, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        """
        *args: data_paths (str, list or tuple): full path/paths to root dir/dirs from where
                            the local file paths must be collected         
                extensions (str or tuple): format of images to be collected ('.jpeg', '.jpg', '.png', etc.)

                root (str): root directory with the resource files
                label_path (str): path to .csv file containing labels
                transform (callable, optional): Optional transform to be applied
                            on a sample.
        """
        self.transform = transform
        self.root_dir = root

        # Create a df from the list of dictionaries in self._found_dataset
        # self_found_dataset contains root in the format
        # D:\..\pytorch_multiproject_vcs\pytorch_multiproject\resources\wiki_crop\00
        # we need to convert this into 00\image.jpg in order to perform subset operation with df containing labels

        names = []
        for name_group in self._found_dataset:
            names.extend([os.path.join(os.path.basename(name_group['root']), name) for name in name_group['names']])

        full_df = full_df.copy()
        full_df['image_path'] = full_df['image_path'].apply(lambda val: os.path.normpath(val))
        subset_df = full_df[full_df['image_path'].isin(names)]
        subset_df.reset_index(inplace=True, drop=True)

        self.old_df = subset_df[subset_df['age_group'] == 'old']
        self.young_df = subset_df[subset_df['age_group'] == 'young']

    def __len__(self):
        min_len = min(len(self.old_df), len(self.young_df))
        return min_len

    def __getitem__(self, item):
        old_name = os.path.join(self.root_dir, self.old_df.iloc[item, 0])

        # second image for pair is selected randomly
        random_young = random.randint(0, len(self.young_df) - 1)
        young_name = os.path.join(self.root_dir,
                                  self.young_df.iloc[random_young, 0])

        old_img = np.array(Image.open(old_name))
        young_img = np.array(Image.open(young_name))

        # Add third channel to grayscale images
        if len(old_img.shape) != 3:
            old_img = np.repeat(old_img[:, :, np.newaxis], 3, axis=2)
        if len(young_img.shape) != 3:
            young_img = np.repeat(young_img[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            old_img = self.transform(old_img)
            young_img = self.transform(young_img)

        return old_img, young_img
