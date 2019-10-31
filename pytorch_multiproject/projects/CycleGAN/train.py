import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(ROOT_DIR)
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from data.cycle_gan_dataset import CycleGanDataset
from models.cycle_GAN import CycleGAN
from models.gan_discriminator import GanDiscriminator
from models.gan_generator import GanGenerator

from logger.logger import main_run, default_log_config
from utils import freeze_unfreeze_model, weights_inint_seq

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    resources_dir = os.path.join(ROOT_DIR, 'resources', 'horse_zebras')
    label_path = os.path.join(ROOT_DIR, 'resources', 'horse_zebras', 'dataset_info.csv')


