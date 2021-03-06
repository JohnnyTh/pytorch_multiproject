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
from models.cycle_GAN import CycleGAN, GanGeneratorSmall, GanDiscriminatorSmall, GanOptimizer, GanLrScheduler
from data.cycle_gan_dataset_small import CycleGanDatasetSmall
from trainers.cycle_gan_trainer import CycleGanTrainer
from torchvision.datasets import MNIST, SVHN
from logger.logger import main_run, default_log_config
from utils import normal_weights

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'small.json'


def main(config, args):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', config['resource_dir'])

    mnist_train = MNIST(os.path.join(resources_dir, 'mnist_train'), download=True, train=True)
    svhn_train = SVHN(os.path.join(resources_dir, 'svhn_train'), download=True, split='train')

    mnist_test = MNIST(os.path.join(resources_dir, 'mnist_test'), download=True, train=False)
    svhn_test = SVHN(os.path.join(resources_dir, 'svhn_test'), download=True, split='test')

    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((32, 32)),
                                        transforms.ToTensor()])

    # get datasets
    train_dataset = CycleGanDatasetSmall(mnist_train, svhn_train, sample_size=2000, transform=trans_non_aug)
    test_dataset = CycleGanDatasetSmall(mnist_test, svhn_test, sample_size=100, transform=trans_non_aug)

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
    epochs = config.get('epochs', 40)

    # define generator
    generator = GanGeneratorSmall(num_resblocks=9, skip_relu=False)
    # define discriminator
    discriminator = GanDiscriminatorSmall()
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

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, (epoch + 1 - 20) / float(20))
        return lr_l

    scheduler_gen = optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=lambda_rule)
    scheduler_disc = optim.lr_scheduler.LambdaLR(optim_disc, lr_lambda=lambda_rule)

    # enable parallel forward pass computation if possible
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = GanOptimizer(optim_gen, optim_disc)
    lr_sched = GanLrScheduler(scheduler_gen, scheduler_disc)

    trainer = CycleGanTrainer(dataloaders=dataloaders, root=ROOT_DIR, model=model, criterion=None, optimizer=optimizer,
                              scheduler=lr_sched, metrics=metrics, epochs=epochs, save_dir=args.save_dir,
                              checkpoint=args.checkpoint, change_lr=args.change_lr)

    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
