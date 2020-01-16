import os
import torch
import numpy as np
from data.generic_dataset import GenericDataset


class Word2VecDataset(GenericDataset):

    def __init__(self, root, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.root = root

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
