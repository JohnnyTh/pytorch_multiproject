import os
import pandas as pd
from torch.utils.data import Dataset

class GenericDataset(Dataset):

    def __init__(self, data_path):
        """
            Instance of the class collects the local paths to files in the provided root dir/dirs

            Arguments:
                :param data_path (str, list or tuple): full path/paths to root dir/dirs from where
                          the file paths must be collected
        """
        self.root_dir = data_path

        data = []
        # When list of data paths is provided
        if isinstance(self.root_dir, (list, tuple)):
            for single_path in self.root_dir:
                for root, dirs, files in os.walk(single_path):
                    for file in files:
                        rel_path = os.path.relpath(root, single_path)
                        data.append(os.path.join(rel_path, file))
        # When singe data path is provided
        elif isinstance(self.root_dir, str):
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    # Infer relative path to the file in file system and record it
                    rel_path = os.path.relpath(root, self.root_dir)
                    data.append(os.path.join(rel_path, file))
        else:
            raise Exception("""Provided data path format not supported, 
            please provide str for single path OR list, tuple of strings for multiple paths""")

        # self.found_dataset contains a list of local paths to found data files
        self.found_dataset = data
