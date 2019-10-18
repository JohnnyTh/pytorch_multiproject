import os
import numpy as np
from skimage import io
from data.generic_dataset import GenericDataset


class AgeGenderDataset(GenericDataset):

    def __init__(self,  full_df, root, *args, transform=None, **kwargs):
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

        # replace linux slash with windows  backslash to check intersection of two sets of image paths
        full_df = full_df.copy()
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
