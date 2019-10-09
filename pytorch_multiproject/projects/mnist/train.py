import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(ROOT_DIR)  # Add root project directory to the current path

import argparse
import gzip
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from models.mnist_model import Nist_Net
from trainers.mnist_trainer import MnistTrainer
from data import MnistDataset, MnistSubset
from torch.utils.data import DataLoader, ConcatDataset
from utils import read_json

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(args):
    # configure the logger
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filename=os.path.join(ROOT_DIR, 'saved', 'mnist', 'log', 'info.log'))
    logger = logging.getLevelName('train')

    # read hyperparameter instructions from json file
    config = read_json(args.config)

    # get the datasets
    mnist_path = os.path.join(ROOT_DIR, 'resources', 'mnist', 'mnist.pkl.gz')
    with gzip.open(mnist_path) as gzip_train:
        mnist_train = MnistDataset(gzip_train, 'train', 28)
    with gzip.open(mnist_path) as gzip_val:
        mnist_val = MnistDataset(gzip_val, 'train', 28)

    indices = np.random.randint(0, len(mnist_train), config['subset_size'])
    mnist_train_transf = MnistSubset(mnist_train, indices,
                                     transform=transforms.Compose([transforms.ToPILImage(),
                                                                   transforms.RandomPerspective(p=1.0),
                                                                   transforms.ToTensor()]))
    mnist_train_concat = ConcatDataset([mnist_train, mnist_train_transf])

    minibatch_size = config['dataloader']['minibatch_size']
    num_workers = config['dataloader']['num_workers']
    data_loader = DataLoader(mnist_train_concat, batch_size=minibatch_size, num_workers=num_workers, shuffle=True)
    val_data_loader = DataLoader(mnist_val, batch_size=minibatch_size, num_workers=num_workers, shuffle=False)

    model = Nist_Net()
    logger.info(model)

    # Define a Loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'])

    # Set a learning rate scheduler
    lambda_ = lambda epoch: config['scheduler_decay_rate'] ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_)

    trainer = MnistTrainer(model, criterion, optimizer,
                           config=config, data_loader=data_loader,
                           val_data_loader=val_data_loader,
                           scheduler=scheduler)
    trainer.initialize_weights()
    # trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convolutional model for MNIST dataset ')
    parser.add_argument('-c', '--config', type=str, default='train.json',
                        help='Config file path')
    args = parser.parse_args()

    main(args)
