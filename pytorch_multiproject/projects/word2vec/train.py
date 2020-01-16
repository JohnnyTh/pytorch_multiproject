import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import torch
from torch.utils.data import DataLoader
from models.word2vec import Word2VecModel
from data.word2vec_dataset import Word2VecDataset
from trainers.word2vec_trainer import Word2VecTrainer
from logger.logger import main_run, default_log_config
import data.custom_transforms as t_custom

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config, args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', config.get('resource_dir', 'cornell movie-dialogs corpus'))

