import os
import numpy as np
from skimage import io
from data.generic_dataset import GenericDataset


class CycleGanDataset(GenericDataset):

    def __init__(self, full_df, root, mode, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        """
        *args: data_paths (str, list or tuple): full path/paths to root dir/dirs from where
                          the local file paths must be collected
               extensions (str or tuple): format of images to be collected ('.jpeg', '.jpg', '.png', etc.)
        root (str): root directory with the resource files
        mode (str): _______
        label_path (str): path to .csv file containing labels
        transform (callable, optional): Optional transform to be applied
                          on a sample.
        """
        self.transform = transform
        self.root_dir = root
        if mode == 'train' or mode == 'test':
            self.mode = mode
        else:
            raise Exception('Provide correct mode name ("train" or "test")')

        names = []
        for name_group in self._found_dataset:
            names.extend([os.path.join(os.path.basename(name_group['root']), name) for name in name_group['names']])

        full_df = full_df.copy()
        subset_df = full_df[full_df['image_path'].isin(names)]
        subset_df.reset_index(inplace=True, drop=True)

        # df['path'] - 'sources|targets/00/example.jpg'
        # df['source_or_target'] - 'source' | 'target'
        if mode == 'train':
            series_sources = subset_df[subset_df['designation'] == 'source_train']['image_path']
            series_targets = subset_df[subset_df['designation'] == 'target_train']['image_path']
        elif mode == 'test':
            series_sources = subset_df[subset_df['designation'] == 'source_test']['image_path']
            series_targets = subset_df[subset_df['designation'] == 'target_test']['image_path']

        series_sources.reset_index(inplace=True, drop=True)
        series_targets.reset_index(inplace=True, drop=True)
        if len(series_sources) != len(series_targets):
            min_len = min(len(series_sources), len(series_targets))
            series_sources = series_sources.loc[:min_len]
            series_targets = series_targets.loc[:min_len]

        self.series_sources = series_sources
        self.series_targets = series_targets

    def __len__(self):
        return len(self.series_sources)

    def __getitem__(self, item):
        """
           Returns unpaired source and target images
        """
        img_name_source = os.path.join(self.root_dir, self.series_sources.iloc[item])
        img_name_target = os.path.join(self.root_dir, self.series_targets.iloc[item])
        img_source = io.imread(img_name_source)
        img_target = io.imread(img_name_target)

        # to deal with grayscale images (1 channel instead of 3)
        if len(img_source.shape) != 3:
            img_source = np.repeat(img_source[:, :, np.newaxis], 3, axis=2)
        if len(img_target.shape) != 3:
            img_source = np.repeat(img_target[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target
