import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(ROOT_DIR)
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from data.gender_age_dataset import AgeGenderDataset
from models.age_gender_model import AgeGenderModelV2
from trainers.age_gender_trainer import AgeGenderTrainer
from logger.logger import main_run, default_log_config

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'small.json'


def main(config, args):
    """
    Args:
    :param config: dictionary with hyperparameters (e.g. number of epochs, learning rate, etc.)
    :param args: optional arguments from argparser
    """

    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    resources_dir = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')
    label_path = os.path.join(ROOT_DIR, 'resources', 'wiki_crop', 'dataset_info.csv')

    # get df with labels data
    dataset_df = pd.read_csv(label_path, usecols=[1, 2, 3])
    dataset_df['gender'] = dataset_df['gender'].astype(float)

    # split the full df into train and test datasets
    train_size = config.get('train_size', 35000)
    test_size = int(train_size * config.get('test_share', 0.18))
    train_df = dataset_df.loc[0: train_size]

    # grab all the remaining from train split data
    test_imbalanced = dataset_df.iloc[train_size:]
    # get all the female and male images
    test_female = test_imbalanced[test_imbalanced['gender'] == 0]
    test_male = test_imbalanced[test_imbalanced['gender'] == 1]
    # get the full balanced dataset based on the length of female dataset
    test_balanced_full = pd.concat((test_female, test_male.iloc[0: len(test_female)]))

    # get a random sample from concatenated balanced df
    if test_size <= len(test_balanced_full):
        test_df = test_balanced_full.sample(test_size)
        test_df.reset_index(drop=True, inplace=True)
    else:
        logger.warning('Please decrease the size of test dataset, the are not enough data. '
                       'The size of test must be below {}'.format(len(test_balanced_full)))
        raise ValueError('Could not create test dataset!')

    # collect list of folders containing input images
    data_dirs = [os.path.join(resources_dir, o)
                 for o in os.listdir(resources_dir)
                 if os.path.isdir(os.path.join(resources_dir, o))]

    # Prepare the Dataset instances. Note that the images are resized for VGG 16
    trans_non_aug = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # get datsets
    train_dataset = AgeGenderDataset(full_df=train_df, root=resources_dir, data_paths=data_dirs,
                                     extensions=(('.jpg'),)*len(data_dirs), transform=trans_non_aug)
    test_dataset = AgeGenderDataset(full_df=test_df, root=resources_dir, data_paths=data_dirs,
                                    extensions=(('.jpg'),)*len(data_dirs), transform=trans_non_aug)

    # create dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                              num_workers=config.get('num_workers', 0))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False,
                                             num_workers=config.get('num_workers', 0))
    dataloaders = {'train': trainloader, 'val': testloader}

    # define metrics
    metrics = {
        'loss': {'gender': 10.0, 'age': 100.0, 'total': 100.0},
        'acc_gender': 0.0
    }
    # define number of epochs
    epochs = config['epochs']

    # Get the pretrained donor model
    resnet = models.resnet18(pretrained=True)
    # Create a model using donor
    resnet_age_gender = AgeGenderModelV2(resnet)

    # Binary cross entropy loss for gender prediction
    criterion_gender = nn.BCELoss()
    # L1 loss-  measures the mean absolute error (MAE) between each element in the input x and target y
    criterion_age = nn.L1Loss()
    criterion = {'gender': criterion_gender, 'age': criterion_age}

    # setting an optimizer
    params = list(resnet_age_gender.classifier_age.parameters()) + list(resnet_age_gender.classifier_gender.parameters())
    optimizer = optim.Adam(params, lr=config['learning_rate'], weight_decay=1e-5)

    # Set a learning rate scheduler
    lambda_ = lambda epoch: config.get('lr_sched_lambda', 0.89) ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_)

    # create a session of trainer
    session = AgeGenderTrainer(dataloaders, ROOT_DIR, resnet_age_gender, criterion, optimizer, scheduler,
                               metrics, epochs, hyperparams=config, save_dir=args.save_dir, checkpoint=args.checkpoint,
                               change_lr=args.change_lr)

    # run the training session
    logger.info('Training session begins.')
    logger.info('Using device {}'.format(torch.cuda.get_device_name(0)))
    session.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
