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
from models.age_gender_model import AgeGenderModel
from trainers.age_gender_trainer import AgeGenderTrainer
from logger.logger import main_run, default_log_config
from utils import freeze_unfreeze_model, weights_inint_seq

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config):
    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    resources_dir = os.path.join(ROOT_DIR, 'resources', 'wiki_crop')
    label_path = os.path.join(ROOT_DIR, 'resources', 'wiki_crop', 'dataset_info.csv')

    # get df with labels data
    dataset_df = pd.read_csv(label_path, usecols=[1, 2, 3])
    dataset_df['gender'] = dataset_df['gender'].astype(float)

    # split the full df into train and test datasets
    train_size = 5000
    test_size = train_size * 0.25
    train_df = dataset_df.loc[0: train_size]
    test_df = dataset_df.loc[train_size: train_size+test_size]

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
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    dataloaders = {'train': trainloader, 'val': testloader}

    # define metrics
    metrics = {
        'loss': {'gender': 10.0, 'age': 100.0, 'total': 100.0},
        'acc_gender': 0.0
    }
    # define number of epochs
    epochs = config['epochs']

    # Create a model template
    vgg11_age_gender = AgeGenderModel()
    # Get the pretrained donor model
    vgg11 = models.vgg11(pretrained=True)
    # Transfer the features' layer parameters from the donor
    vgg11_age_gender.features = vgg11.features
    freeze_unfreeze_model(vgg11_age_gender, 'freeze')
    # Determine the number of features passed from the last  layer of the frozen feature extractor
    in_features = vgg11.classifier[0].in_features

    # get new heads for gender and age
    sequential_gender = vgg11_age_gender.gender_head(in_features)
    sequential_age = vgg11_age_gender.age_head(in_features)

    # initialize the weights of new layers using Xavier init
    weights_inint_seq(sequential_gender)
    weights_inint_seq(sequential_age)

    # attach new heads to the model
    vgg11_age_gender.classifier_gender = sequential_gender
    vgg11_age_gender.classifier_age = sequential_age

    # Binary cross entropy loss for gender prediction
    criterion_gender = nn.BCELoss()
    # L1 loss-  measures the mean absolute error (MAE) between each element in the input x and target y
    criterion_age = nn.L1Loss()
    criterion = {'gender': criterion_gender, 'age': criterion_age}

    # setting an optimizer
    params = list(vgg11_age_gender.classifier_age.parameters()) + list(vgg11_age_gender.classifier_gender.parameters())
    optimizer = optim.Adam(params, lr=config['learning_rate'], weight_decay=1e-5)

    # Set a learning rate scheduler
    lambda_ = lambda epoch: 0.89 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_)

    # create a session of trainer
    trainer = AgeGenderTrainer(dataloaders, scheduler, ROOT_DIR, vgg11_age_gender,
                               criterion, optimizer, metrics, epochs)

    # run the training session
    logger.info('Training session begins.')
    logger.info('Using device {}'.format(torch.cuda.get_device_name(0)))
    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
