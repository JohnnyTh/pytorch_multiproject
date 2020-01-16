import os
import sys
import math
import logging
import torch
from trainers.generic_trainer import GenericTrainer
from tqdm import tqdm


class Word2VecTrainer(GenericTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _train_step(self, epoch):
        pass