import os
import sys
import math
import logging
import torch
import torchvision
from tqdm import tqdm
from trainers.generic_trainer import GenericTrainer
from utils import warmup_lr_scheduler, reduce_dict
from utils import get

class MaskRCNNTrainer(GenericTrainer):

    def __init__(self, dataloaders, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(os.path.basename(__file__))
        self.dataloaders = dataloaders
        # create directory for saving the results of val phase
        self.save_dir_test = os.path.join(self.save_dir, 'gan_test')
        if not os.path.exists(self.save_dir_test):
            os.mkdir(self.save_dir_test)

    def _train_step(self, epoch):
        """Behaviour during one pass through the epoch."""
        self.logger.info('Epoch {}/{}'.format(epoch, self.epochs))
        self.logger.info('-' * 10)

        # print parameters of optimizer and scheduler every epoch
        self.logger.info(str(self.optimizer))
        if self.scheduler is not None:
            self.logger.info(str(self.scheduler))

        results = {
            'best_performance': False
        }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                self.model.train()  # Set model to training mode

                lr_scheduler_warmup = None
                if epoch == 0:
                    warmup_factor = 1. / 1000
                    warmup_iters = min(1000, len(self.dataloaders[phase]) - 1)
                    lr_scheduler_warmup = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

            else:
                self.model.eval()  # Set model to evaluation mode
                coco = get_coco_api_from_dataset(data_loader.dataset)
                iou_types = _get_iou_types(model)
                coco_evaluator = CocoEvaluator(coco, iou_types)

            t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
            for images, targets in t:
                if phase == 'train':
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets)

                    losses = sum(loss for loss in loss_dict.values())

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                    loss_value = losses_reduced.item()

                    if not math.isfinite(loss_value):
                        self.logger.info("Loss is {}, stopping training".format(loss_value))
                        self.logger.info(loss_dict_reduced)
                        sys.exit(1)

                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    if lr_scheduler_warmup is not None:
                        lr_scheduler_warmup.step()

                elif phase == 'val':
                    with torch.set_grad_enabled(False):


            if phase == 'train':
                self.scheduler.step()


        return results

    def _get_iou_types(model):
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types