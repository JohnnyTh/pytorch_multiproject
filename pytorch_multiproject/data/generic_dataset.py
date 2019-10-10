import os
import pandas as pd
from torch.utils.data import Dataset

class GenericDataset(Dataset):

    def __init__(self, data_path, target_dtype, label_path):

        self.root_dir = data_path
        self.supported_data_format = {'img': ('.jpg', '.jpeg', '.png'), 'text': '.txt'}
        assert target_dtype in self.supported_data_format.keys(), "Provide supported dtype name ('img', 'text')!"

        data = []
        # When list of data paths is provided
        if isinstance(self.root_dir, (list, tuple)):
            for entry in self.root_dir:
                for root, dirs, files in os.walk(entry):
                    for file in files:
                        if file.endswith(self.supported_data_format[target_dtype]):
                            rel_path = os.path.relpath(root, entry)
                            data.append(os.path.join(rel_path, file))
        # When singe data path is provided
        elif isinstance(self.root_dir, str):
            for root, dirs, files in os.walk(self.root_dir):
                for file in files:
                    if file.endswith(self.supported_data_format[target_dtype]):
                        # Infer relative path to the file in file system and record it
                        rel_path = os.path.relpath(root, self.root_dir)
                        data.append(os.path.join(rel_path, file))
        else:
            raise Exception("""Provided data path format not supported, 
            please provide str for single path OR list, tuple of strings for multiple paths""")

        # self.dataset contains a list of local paths to data files
        self.dataset = data