import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
from skimage import io
from generic_dataset import GenericDataset


class AgeGenderDataset(GenericDataset):

    def __init__(self, data_path, target_dtype, label_path, transform=None):
        super(AgeGenderDataset, self).__init__(data_path, target_dtype, label_path)
        self.transform = transform

        full_df = pd.read_csv(label_path, usecols=[1, 2, 3])
        # replace windows slash with linux backslash to check intersection of two sets of image paths
        subset = [data_entry.replace("\\", "/") for data_entry in self.dataset]
        subset_df = full_df[full_df['image_path'].isin(subset)]

        if os.name == 'nt':
            # Change the slash back if running windows system
            subset_df['image_path'] = subset_df['image_path'].str.replace('/', "\\")

        self.dataframe = subset_df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        img = io.imread(img_name)
        label_gender = self.dataframe.iloc[idx, 1]
        label_age = self.dataframe.iloc[idx, 2]

        # Add third channel to grayscale images (required for VGG 16)
        if len(img.shape) != 3:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            img = self.transform(img)

        return img, label_gender, label_age

    def get_all_labels(self):
        return self.dataframe.iloc[:, [1, 2]]


data_dir = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')
labels = os.path.join(ROOT_DIR, 'resources', 'wiki_crop', 'dataset_info.csv')

test_dataset = AgeGenderDataset(data_dir, 'img', labels)