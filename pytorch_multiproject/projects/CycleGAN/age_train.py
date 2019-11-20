import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from models.cycle_GAN import CycleGAN, GanGenerator, GanDiscriminator, GanOptimizer, GanLrScheduler
from data.cycle_gan_age_dataset import AgeGanDataset
from trainers.cycle_gan_trainer import CycleGanTrainer
from logger.logger import main_run, default_log_config
from utils import normal_weights
from data import Denormalize

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'age.json'


def main(config, args):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')
    label_path = os.path.join(ROOT_DIR, 'resources', 'wiki_crop', 'dataset_info.csv')

    # get df with labels data
    dataset_df = pd.read_csv(label_path, usecols=[1, 2, 3])
    dataset_df['gender'] = dataset_df['gender'].astype(float)

    # separate the dataset into old and young subsets
    old_df = dataset_df[dataset_df['age'] > 60]
    young_df = dataset_df[dataset_df['age'] < 30]

    # balance dfs based on genders (females are minority)
    old_g = old_df.groupby('gender')
    old_df_balanced = old_g.apply(lambda x: x.sample(old_g.size().min())).reset_index(drop=True)

    young_g = young_df.groupby('gender')
    young_df_balanced = young_g.apply(lambda x: x.sample(young_g.size().min())).reset_index(drop=True)

    train_size = 1000
    test_size = 100

    # add young/old labels in order to split the concatenated dataset
    # inside AgeGanDataset
    old_df_balanced['age_group'] = 'old'
    young_df_balanced['age_group'] = 'young'

    train_old = old_df_balanced[: train_size]
    test_old = old_df_balanced[train_size: train_size + test_size]

    # train_young can be much bigger than train_old since the iterations
    # through the image pairs will be performed until reaching the length of
    # smallest dataset
    train_young = young_df_balanced[: -test_size]
    test_young = young_df_balanced[-test_size:]

    train_df = pd.concat((train_old, train_young))
    test_df = pd.concat((test_old, test_young))

    # collect list of folders containing input images
    data_dirs = [os.path.join(resources_dir, o)
                 for o in os.listdir(resources_dir)
                 if os.path.isdir(os.path.join(resources_dir, o))]

    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # get datasets
    train_dataset = AgeGanDataset(full_df=train_df, root=resources_dir,
                                  data_paths=data_dirs,
                                  extensions=(('.jpg'),)*len(data_dirs),
                                  transform=trans_non_aug)

    test_dataset = AgeGanDataset(full_df = test_df, root=resources_dir,
                                 data_paths=data_dirs,
                                 extensions=(('.jpg'),)*len(data_dirs),
                                 transform=trans_non_aug)

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
    # code to run the model


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
