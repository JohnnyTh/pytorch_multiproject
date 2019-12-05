import os
import sys
import math
import logging
import torch
import torchvision
from trainers.generic_trainer import GenericTrainer
from utils import warmup_lr_scheduler
from torchvision.utils import save_image
from tqdm import tqdm
# import time
# from utils import reduce_dict, MetricLogger, SmoothedValue
# from utils.coco_utils import get_coco_api_from_dataset
# from utils.coco_eval import CocoEvaluator


class MaskRCNNTrainer(GenericTrainer):

    def __init__(self, dataloaders, val_phase_freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # to supress debug messages from PIL module
        self.pil_logger = logging.getLogger('PIL')
        self.pil_logger.setLevel(logging.INFO)
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.dataloaders = dataloaders

        assert val_phase_freq != 0, 'val phase frequency must be 1 or higher, or None'
        self.val_phase_freq = val_phase_freq

        # create directory for saving the results of val phase
        self.save_dir_test = os.path.join(self.save_dir, 'mask_r_cnn_test')
        if not os.path.exists(self.save_dir_test):
            os.mkdir(self.save_dir_test)

    def _train_step(self, epoch):
        """Behaviour during one pass through the epoch."""

        # print parameters of optimizer and scheduler every epoch
        self.logger.info(str(self.optimizer))
        if self.scheduler is not None:
            self.logger.info(str(self.scheduler.state_dict()))

        results = {
            'best_performance': False
        }

        self.logger.info('Epoch {}/{}'.format(epoch, self.epochs))
        self.logger.info('-' * 10)

        # execute training and validation
        self.train_one_epoch(epoch)

        # execute val phase if conditions are met
        if (self.val_phase_freq is not None
                and epoch % self.val_phase_freq == 0):
            self.val_one_epoch(epoch)

        # in the current implementation we save the trained model after each epoch
        results.update({'best_performance': True})

        return results

    def train_one_epoch(self, epoch):

        self.logger.info('Starting train phase')
        # print training results every x iterations
        print_freq = 10
        self.model.train()
        phase = 'train'

        # metric_logger = MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler_warmup = None
        # the first epoch is 1 not 0
        if epoch == 1:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.dataloaders[phase]) - 1)
            lr_scheduler_warmup = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
        for idx, data in enumerate(t):
            images, targets = data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                self.logger.info("Loss is {}, stopping training".format(loss_value))
                self.logger.info(loss_dict)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if lr_scheduler_warmup is not None:
                lr_scheduler_warmup.step()

            if idx % print_freq == 0:
                loss_dict_str = {key: '{:.6f}'.format(value) for key, value in loss_dict.items()}
                print_data = [epoch, self.epochs, self.optimizer.param_groups[0]["lr"], loss_value, loss_dict_str]
                self.logger.info('Epoch {}/{}   optim_lr: {:.6f},   loss: {:.6f}    {}'.format(*print_data))

            # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            # metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        self.scheduler.step()

    @torch.no_grad()
    def val_one_epoch(self, epoch):
        self.logger.info('Starting val phase')

        self.model.eval()  # Set model to evaluation mode
        phase = 'val'

        # n_threads = torch.get_num_threads()
        # torch.set_num_threads(1)
        cpu_device = torch.device("cpu")

        # metric_logger = MetricLogger(delimiter="  ")
        # header = 'Test:'

        # coco = get_coco_api_from_dataset(self.dataloaders[phase].dataset)
        # iou_types = self._get_iou_types(self.model)
        # coco_evaluator = CocoEvaluator(coco, iou_types)

        t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
        for image, targets in t:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            image = list(img.to(device) for img in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # torch.cuda.synchronize()
            # model_time = time.time()
            outputs = self.model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

            # evaluator_time = time.time()
            # coco_evaluator.update(res)
            # evaluator_time = time.time() - evaluator_time
            # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        # print("Averaged stats:", metric_logger)
        # coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        # torch.set_num_threads(n_threads)

    @torch.no_grad()
    def test(self, num_masks=5):
        self.model.eval()

        # metric_logger = MetricLogger(delimiter="  ")
        # header = 'Test:'
        cpu_device = torch.device("cpu")

        t = tqdm(iter(self.dataloaders), leave=False, total=len(self.dataloaders))
        for idx, data in enumerate(t):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # data[0] - images, data[1] - targets
            image = list(img.to(device) for img in data[0])

            torch.cuda.synchronize()
            outputs = self.model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            all_masks = outputs[0]['masks']
            # unpack the top num_masks of shape N, 1, H, W into a list of 1, H, W masks
            top_masks = [*all_masks[: num_masks]]

            to_save = {'source_img': image[0], 'mask': top_masks}

            # script that saves the source img and top generated masks
            for key in to_save.keys():
                if key == 'source_img':
                    save_image(to_save[key], os.path.join(self.save_dir_test, key+'_{}.png'.format(idx)))
                elif key == 'mask':
                    for num_mask, mask in enumerate(to_save[key]):
                        save_image(mask, os.path.join(self.save_dir_test, key+'_{}_{}.png'.format(idx, num_mask)))

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
