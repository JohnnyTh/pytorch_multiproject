import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import numpy as np
import pandas as pd
from skimage import io
from generic_dataset import GenericDataset


class AgeGenderDataset(GenericDataset):

    def __init__(self, data_path, extensions, label_path, root, transform=None):
        """
        :param data_path (str, list or tuple): full path/paths to root dir/dirs from where
                          the local file paths must be collected
        :param target_dtype (str or tuple): format of images to be collected ('.jpeg', '.jpg', '.png', etc.)
        :param label_path (str): path to .csv file containing labels
        :param transform (callable, optional): Optional transform to be applied
                          on a sample.
        """

        super(AgeGenderDataset, self).__init__(data_path, extensions)
        self.transform = transform
        self.root_dir = root

        full_df = pd.read_csv(label_path, usecols=[1, 2, 3])

        # Create a df from the list of dictionaries in self._found_dataset
        # self_found_dataset contains root in format
        # D:\..\pytorch_multiproject_vcs\pytorch_multiproject\resources\wiki_crop\00
        # we need to convert this into 00\image.jpg in order to perform subset operation with df containing labels

        names = []
        for name_group in self._found_dataset:
            names.extend([os.path.join(os.path.basename(name_group['root']), name) for name in name_group['names']])

        # replace linux slash with windows  backslash to check intersection of two sets of image paths
        full_df['image_path'] = full_df['image_path'].apply(lambda val: os.path.normpath(val))
        subset_df = full_df[full_df['image_path'].isin(names)]
        subset_df.reset_index(inplace=True, drop=True)

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


"""Some code for testing"""

data_root = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')
data_dirs = [os.path.join(data_root, o) for o in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, o))]
extensions = (('.jpg', '.png', '.jpeg'), )*len(data_dirs)
labels = os.path.join(ROOT_DIR, 'resources', 'wiki_crop', 'dataset_info.csv')

test_dataset = AgeGenderDataset(data_dirs, extensions, labels, data_root)
# print(test_dataset[160])