import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from models.mask_r_cnn_model import get_mask_r_cnn
from data.mask_r_cnn_dataset import PennFudanDataset
from trainers.mask_r_cnn_trainer import MaskRCNNTrainer
from logger.logger import main_run, default_log_config
from utils import collate_fn
import data.custom_transforms as t_custom

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'
torch.manual_seed(0)


def main(config, args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', config.get('resource_dir', 'PennFudanPed'))

    images = os.path.join(resources_dir, 'PNGImages')
    masks = os.path.join(resources_dir, 'PedMasks')

    transform_1 = t_custom.Compose([t_custom.ToTensor(),
                                    t_custom.RandomHorizontalFlip(0.5)])
    transform_2 = t_custom.Compose([t_custom.RandomCropBbox(),
                                    t_custom.ToTensor()])
    transform_3 = t_custom.Compose([t_custom.ColorJitterBbox(0.5, 0.5, 0.5, 0.5),
                                    t_custom.ToTensor()])

    dataset_1 = PennFudanDataset(root=resources_dir, data_paths=[images, masks], extensions=(('.png'),) * 2,
                               transforms=transform_1)
    dataset_2 = PennFudanDataset(root=resources_dir, data_paths=[images, masks], extensions=(('.png'),) * 2,
                                   transforms=transform_2)
    dataset_3 = PennFudanDataset(root=resources_dir, data_paths=[images, masks], extensions=(('.png'),) * 2,
                                 transforms=transform_3)
    dataset_test = PennFudanDataset(root=resources_dir, data_paths=[images, masks], extensions=(('.png'),) * 2,
                                    transforms=t_custom.ToTensor())

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset_1)).tolist()

    dataset_train_aug_1 = Subset(dataset_1, indices[:-50])
    dataset_train_aug_2 = Subset(dataset_2, indices[:-50])
    dataset_train_aug_3 = Subset(dataset_3, indices[:-50])
    dataset_train = ConcatDataset([dataset_train_aug_1, dataset_train_aug_2, dataset_train_aug_3])

    dataset_test = Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    dataloaders = {'train': data_loader, 'val': data_loader_test}

    model = get_mask_r_cnn(num_classes=2)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.get('lr', 0.005), momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    trainer = MaskRCNNTrainer(dataloaders=dataloaders, root=ROOT_DIR, model=model, criterion=None,
                              optimizer=optimizer, scheduler=lr_scheduler, metrics={}, epochs=config.get('epochs', 10),
                              save_dir=args.save_dir, checkpoint=args.checkpoint, change_lr=args.change_lr)
    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
