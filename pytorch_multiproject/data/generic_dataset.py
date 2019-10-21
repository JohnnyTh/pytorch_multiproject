import os
from collections.abc import Iterable
from torch.utils.data import Dataset


class GenericDataset(Dataset):

    def __init__(self, data_paths, extensions):
        """
            Instance of the class collects the local paths to files in the provided root dir/dirs

            Arguments:
                :param data_path (sequence): full path/paths to root dir/dirs from where
                          the file paths must be collected.
                :param extensions (sequence): a sequence of target file extensions corresponding
                          to each entry of data_path.
        """

        self._found_dataset = []

        assert isinstance(data_paths, Iterable) and not isinstance(data_paths, str), 'Check datatype'
        assert isinstance(extensions, Iterable) and not isinstance(extensions, str), 'Check datatype'

        if len(data_paths) > 1:
            if 1 < len(extensions) != len(data_paths):
                raise Exception('Wrong configuration of sources')
        if len(data_paths) == 1 and len(extensions) > len(data_paths):
            data_paths = data_paths*len(extensions)

        for (_dir, _ext) in zip(data_paths, extensions):
            self._found_dataset.append({'root': _dir, 'names': [name for name in os.listdir(_dir) if name.endswith(_ext)]})

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def get_dataset(self):
        return self._found_dataset

