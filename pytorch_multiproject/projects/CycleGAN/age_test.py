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
from data import Denormalize

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'age.json'
random_seed = 1


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

    # shuffle the rows before taking train and test samples
    old_df_balanced = old_df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    young_df_balanced = young_df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    test_old = old_df_balanced[train_size: train_size + test_size]
    test_young = young_df_balanced[-test_size:]

    test_df = pd.concat((test_old, test_young))

    # collect list of folders containing input images
    data_dirs = [os.path.join(resources_dir, o)
                 for o in os.listdir(resources_dir)
                 if os.path.isdir(os.path.join(resources_dir, o))]

    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = AgeGanDataset(full_df=test_df, root=resources_dir,
                                 data_paths=data_dirs,
                                 extensions=(('.jpg'),)*len(data_dirs), random_pairs=False,
                                 transform=trans_non_aug)

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

    trainer = CycleGanTrainer(dataloaders=test_loader, denorm=Denormalize(),
                              root=ROOT_DIR, model=model, criterion=None, optimizer=optimizer,
                              scheduler=None, metrics=None, epochs=1,
                              save_dir=args.save_dir, checkpoint=args.checkpoint)

    trainer.test()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
