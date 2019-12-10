import os
import sys
import math
import logging
import torch
import torchvision
from trainers.generic_trainer import GenericTrainer
from utils import warmup_lr_scheduler
from utils.detection_evaluator import DetectionEvaluator
from utils.mask_saver import MaskSaver
from torchvision.utils import save_image
from tqdm import tqdm


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
        # print training results every x iterations
        print_freq = 10
        phase = 'train'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.info('Starting train phase')
        self.model.train()

        lr_scheduler_warmup = None
        # the first epoch is 1 not 0
        if epoch == 1:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.dataloaders[phase]) - 1)
            lr_scheduler_warmup = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
        for idx, data in enumerate(t):
            images, targets = data

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

        self.scheduler.step()

    @torch.no_grad()
    def val_one_epoch(self, epoch):
        phase = 'val'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        evaluator = DetectionEvaluator()
        mask_saver = MaskSaver(save_dir=self.save_dir_test)

        self.logger.info('Starting val phase')
        self.model.eval()  # Set model to evaluation mode

        t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
        for images, targets in t:
            images = list(image.to(device) for image in images)
            outputs = self.model(images)

            # move results for evaluation to cpu
            targets = {k: v.to('cpu') for k, v in targets[0].items()}
            outputs = {k: v.to('cpu') for k, v in outputs[0].items()}

            save_img = images[0].mul(255).permute(1, 2, 0).byte().numpy()
            masks = outputs['masks'].mul(255).byte().numpy()

            # collect the results from one iteration here
            evaluator.accumulate(targets, outputs)
            mask_saver.accumulate(save_img, masks)

        # compute the mAP summary here
        iou_threshold = 0.5
        mean_avg_precision = evaluator.bbox_score(iou_threshold=iou_threshold)
        self.logger.info('Mean average precision with IoU threshold {}: {}'.format(iou_threshold, mean_avg_precision))

        # generate and save masked images
        mask_saver.generate_masked_img(mask_draw_precision=0.4, opacity=0.4)
        self.logger.info('Masked image have been saved to {}'.format(self.save_dir_test))

    @torch.no_grad()
    def test(self, num_masks=5):
        cpu_device = torch.device("cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.eval()

        t = tqdm(iter(self.dataloaders), leave=False, total=len(self.dataloaders))
        for idx, data in enumerate(t):
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
