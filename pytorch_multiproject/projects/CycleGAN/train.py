import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import itertools
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from models.cycle_GAN import CycleGAN, GanGenerator, GanDiscriminator, GanOptimizer, GanLrScheduler
from data.cycle_gan_dataset import CycleGanDataset

from trainers.cycle_gan_trainer import CycleGanTrainer

from logger.logger import main_run, default_log_config
from utils import freeze_unfreeze_model, normal_weights

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    resources_dir = os.path.join(ROOT_DIR, 'resources', 'horse2zebra')
    label_path = os.path.join(ROOT_DIR, 'resources', 'dataset_info.csv')

    sources = os.path.join(resources_dir, 'trainA')
    targets = os.path.join(resources_dir, 'trainB')

    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_df = pd.read_csv(label_path)
    dataset = CycleGanDataset(full_df=dataset_df, root=resources_dir, mode='train',
                              data_paths=[sources, targets], extensions=(('.jpg'),)*2, transform=trans_non_aug)


    # create dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # initialize metrics with very high loss values
    metrics = {
        'loss_gen': 100.0,
        'ab_disc_loss': 100.0,
        'ba_disc_loss': 100.0
    }
    # define number of epochs
    epochs = config['epochs']

    # define generator
    generator = GanGenerator(skip_relu=False)
    # define discriminator
    discriminator = GanDiscriminator()
    # define criteria for losses
    gan_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    model_hyperparams = {'lambda_identity': 0.5, 'lambda_a': 10.0, 'lambda_b': 10.0}
    model = CycleGAN(generator, discriminator, gan_loss, cycle_loss, identity_loss, model_hyperparams)

    # initialize with normal weights
    model.apply(normal_weights)

    # create optimizers for generators and discriminators
    optim_gen, optim_disc = model.get_optims(lr=0.0002)

    sched_gen = optim.lr_scheduler.StepLR(optim_gen, step_size=30, gamma=0.1)
    sched_disc = optim.lr_scheduler.StepLR(optim_disc, step_size=30, gamma=0.1)

    # enable parallel forward pass computation if possible
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = GanOptimizer(optim_gen, optim_disc)
    lr_sched = GanLrScheduler(sched_gen, sched_gen)

    session = CycleGanTrainer(dataloader=loader, root=ROOT_DIR, model=model, criterion=None, optimizer=optimizer,
                              scheduler=lr_sched, metrics=metrics, epochs=epochs)

    session.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
