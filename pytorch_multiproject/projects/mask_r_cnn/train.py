import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim
from models.mask_r_cnn_model import get_mask_r_cnn
from data.mask_r_cnn_dataset import PennFudanDataset
from trainers.mask_r_cnn_trainer import MaskRCNNTrainer
from logger.logger import main_run, default_log_config
from utils import collate_fn

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config, args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', config['resource_dir'])

    images = os.path.join(resources_dir, 'PNGImages')
    masks = os.path.join(resources_dir, 'PedMasks')

    dataset = PennFudanDataset(root=resources_dir, data_paths=[images, masks], extensions=(('.png'), )*2,
                               transforms=transforms.ToTensor())

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    model = get_mask_r_cnn(num_classes=2)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    trainer = MaskRCNNTrainer()
    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)