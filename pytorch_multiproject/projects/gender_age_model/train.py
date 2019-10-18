import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(ROOT_DIR)
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data.gender_age_dataset import AgeGenderDataset
from projects.gender_age_model.age_gender_model import AgeGenderModel
from trainers.age_gender_trainer import AgeGenderTrainer
from logger.logger import main_run, default_log_config

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))

    resources_dir = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')
    label_path = os.path.join(ROOT_DIR, 'resources', 'wiki_crop', 'dataset_info.csv')

    dataset_df = pd.read_csv(label_path, usecols=[1, 2, 3])
    dataset_df['gender'] = dataset_df['gender'].astype(float)

    # split the full df into train and test datasets
    train_size = 5000
    test_size = train_size * 0.25
    train_df = dataset_df.loc[0: train_size]
    test_df = dataset_df.loc[train_size: train_size+test_size]

    data_dirs = [os.path.join(resources_dir, o)
                 for o in os.listdir(resources_dir)
                 if os.path.isdir(os.path.join(resources_dir, o))]

    train_dataset = AgeGenderDataset(full_df=train_df, root=ROOT_DIR,
                                     data_paths=data_dirs, extensions=(('.jpg'),)*len(data_dirs))
    test_dataset = AgeGenderDataset(full_df=test_df, root=ROOT_DIR, data_paths=data_dirs,
                                    extensions=(('.jpg'),)*len(data_dirs))



if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
