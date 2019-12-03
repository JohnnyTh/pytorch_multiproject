import os
import sys
import math
import time
import logging
import torch
import torchvision
from trainers.generic_trainer import GenericTrainer
from utils import warmup_lr_scheduler, reduce_dict, MetricLogger, SmoothedValue
from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator


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

        # print parameters of optimizer and scheduler every epoch
        self.logger.info(str(self.optimizer))
        if self.scheduler is not None:
            self.logger.info(str(self.scheduler))

        # in the current implementation we don't save the trained model
        results = {
            'best_performance': False
        }

        self.train_one_epoch(epoch)
        self.val_one_epoch(epoch)

        return results

    def train_one_epoch(self, epoch):
        self.model.train()  # Set model to training mode
        phase = 'train'

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler_warmup = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.dataloaders[phase]) - 1)
            lr_scheduler_warmup = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        for images, targets in metric_logger.log_every(self.dataloaders[phase], 10, header):
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

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        self.scheduler.step()

    @torch.no_grad()
    def val_one_epoch(self, epoch):
        self.model.eval()  # Set model to evaluation mode
        phase = 'val'

        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")

        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(self.dataloaders[phase].dataset)
        iou_types = self._get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for image, targets in metric_logger.log_every(self.dataloaders[phase], 100, header):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            image = list(img.to(device) for img in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator

    @staticmethod
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