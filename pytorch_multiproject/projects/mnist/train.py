import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(ROOT_DIR)
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.mnist_model import MnistNet
from trainers.mnist_trainer import MnistTrainer
from logger.logger import main_run, default_log_config

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'small.json'


def main(config):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))

    resources_dir = os.path.join(ROOT_DIR, 'resources', 'mnist')
    # load datasets
    mnist_trainset = datasets.MNIST(root=resources_dir, train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root=resources_dir, train=False, download=True, transform=transforms.ToTensor())
    # create dataloaders
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=128, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=128, shuffle=False, num_workers=0)
    dataloaders = {'train': trainloader, 'val': testloader}

    # define metrics
    metrics = {'epoch': 0, 'loss': 10., 'acc': 0.0}
    # define number of epochs
    epochs = config['epochs']
    # create an instance of model
    model = MnistNet()
    # define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def f(epoch): return 0.89 ** epoch  # Set a learning rate scheduler

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    trainer = MnistTrainer(dataloaders, scheduler, ROOT_DIR, model, criterion, optimizer, metrics, epochs)

    # run the training session
    logger.info('Training session begins.')
    logger.info('Using device {}'.format(torch.cuda.get_device_name(0)))
    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)

