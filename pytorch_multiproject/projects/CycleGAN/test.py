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


# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config, args):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    resources_dir = os.path.join(ROOT_DIR, 'resources', 'horse2zebra')

    test_sources = os.path.join(resources_dir, 'testA')
    test_targets = os.path.join(resources_dir, 'testB')

    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # get datasets
    test_dataset = CycleGanDataset(root=resources_dir, data_paths=[test_sources, test_targets],
                                   extensions=(('.jpg'),)*2, random_pairs=False, transform=trans_non_aug)

    # create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

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

    # create optimizers for generators and discriminators
    optim_gen, optim_disc = model.get_optims(lr=config.get('lr', 0.0002))

    # enable parallel forward pass computation if possible
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = GanOptimizer(optim_gen, optim_disc)

    trainer = CycleGanTrainer(dataloaders=test_loader, root=ROOT_DIR, model=model, criterion=None, optimizer=optimizer,
                              scheduler=None, metrics=None, epochs=1,
                              save_dir=args.save_dir, checkpoint=args.checkpoint)

    trainer.test()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)