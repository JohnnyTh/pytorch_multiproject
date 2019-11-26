import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from models.cycle_GAN import CycleGAN, GanGenerator, GanDiscriminator, GanOptimizer, GanLrScheduler
from data.cycle_gan_dataset import CycleGanDataset
from trainers.cycle_gan_trainer import CycleGanTrainer
from logger.logger import main_run, default_log_config
from utils import normal_weights
from data import Denormalize


# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config, args):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', config['resource_dir'])

    train_sources = os.path.join(resources_dir, 'trainA')
    train_targets = os.path.join(resources_dir, 'trainB')
    test_sources = os.path.join(resources_dir, 'testA')
    test_targets = os.path.join(resources_dir, 'testB')

    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # get datasets
    train_dataset = CycleGanDataset(root=resources_dir, data_paths=[train_sources, train_targets],
                                    extensions=(('.jpg'),)*2, transform=trans_non_aug)
    test_dataset = CycleGanDataset(root=resources_dir, data_paths=[test_sources, test_targets],
                                   extensions=(('.jpg'),)*2, transform=trans_non_aug)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    dataloaders = {'train': train_loader, 'val': test_loader}

    # initialize metrics with very high loss values so that the first iteration of model always overrides them
    metrics = {
        'loss_gen': 100.0,
        'ab_disc_loss': 100.0,
        'ba_disc_loss': 100.0
    }
    # define number of epochs
    epochs = config.get('epochs', 200)

    # define generator
    generator = GanGenerator()
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
    optim_gen, optim_disc = model.get_optims(lr=config.get('lr', 0.0002))

    # Lr is static during the first 100 epochs and linearly decays until zero over epochs 100-200
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, (epoch + 1 - 100) / float(100))
        return lr_l

    scheduler_gen = optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=lambda_rule)
    scheduler_disc = optim.lr_scheduler.LambdaLR(optim_disc, lr_lambda=lambda_rule)

    # enable parallel forward pass computation if possible
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = GanOptimizer(optim_gen, optim_disc)
    lr_sched = GanLrScheduler(scheduler_gen, scheduler_disc)

    trainer = CycleGanTrainer(dataloaders=dataloaders, denorm=Denormalize(), root=ROOT_DIR, model=model,
                              criterion=None, optimizer=optimizer, scheduler=lr_sched, metrics=metrics,
                              epochs=epochs, save_dir=args.save_dir, checkpoint=args.checkpoint,
                              change_lr=args.change_lr)

    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
