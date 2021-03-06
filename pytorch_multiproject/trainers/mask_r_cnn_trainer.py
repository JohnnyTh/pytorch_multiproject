import os
import sys
import math
import logging
import torch
from datetime import datetime
from trainers.generic_trainer import GenericTrainer
from utils import warmup_lr_scheduler
from utils.detection_evaluator import DetectionEvaluator
from torchvision.utils import save_image
from tqdm import tqdm


class MaskRCNNTrainer(GenericTrainer):

    def __init__(self, dataloaders, val_phase_freq=1, *args, **kwargs):
        """
        Trains and evaluates Mask R-CNN model from torchvision package.
        Parameters
        ----------
        dataloaders (dict): 'train' and 'val' dataloaders.
        val_phase_freq (int, optional): determines how often validation phase is executed. if freq == 1 - every phase.
        *args: root, model, criterion, optimizer, scheduler, metrics, epochs,
               hyperparams (optional), save_dir (optional), checkpoint (optional),
               change_lr (optional).
        **kwargs: checkpoint (optional).
        """
        super().__init__(*args, **kwargs)

        # to suppress debug messages from PIL module
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

        self.log_name = 'mask_rcnn_log_{}.txt'.format(datetime.now().strftime("%Y-%m-%d_%H:%M"))
        self.write_log('epoch avg_precision recall\n')

    def _train_step(self, epoch):
        """Behaviour during one pass through the epoch.
        Parameters
        ----------
        epoch (int): current epoch number.
        """

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
        """
        Parameters
        ----------
        epoch (int): current epoch number.
        """
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
                loss_dict_str = {key: '{:.6f}'.format(value.item()) for key, value in loss_dict.items()}
                print_data = [epoch, self.epochs, self.optimizer.param_groups[0]["lr"], loss_value, loss_dict_str]
                self.logger.info('Epoch {}/{}   optim_lr: {:.6f},   loss: {:.6f}    {}'.format(*print_data))

        self.scheduler.step()

    @torch.no_grad()
    def val_one_epoch(self, epoch):
        """
        Parameters
        ----------
        epoch (int): current epoch number.
        """
        phase = 'val'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        evaluator = DetectionEvaluator(save_dir=self.save_dir_test)

        self.logger.info('Starting val phase')
        self.model.eval()  # Set model to evaluation mode

        t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
        for images, targets in t:
            images = list(image.to(device) for image in images)
            outputs = self.model(images)

            # move results for evaluation to cpu
            targets = {k: v.to('cpu') for k, v in targets[0].items()}
            outputs = {k: v.to('cpu') for k, v in outputs[0].items()}

            # denormalize image, convert its shape into H, W, C to prepare it for PIL manipulations
            save_img = images[0].mul(255).permute(1, 2, 0).byte().cpu().numpy()

            # collect the results from one iteration here
            evaluator.accumulate(save_img, targets, outputs)

        # compute the mAP summary here
        iou_threshold = 0.75
        non_max_iou_thresh = 0.4
        score_thresh = 0.6
        avg_precision, precision, recall, selected_boxes = evaluator.bbox_score(
            iou_threshold=iou_threshold,
            non_max_iou_thresh=non_max_iou_thresh,
            score_threshold=score_thresh)

        self.logger.info('Average precision with IoU threshold {}: {}'.format(iou_threshold, avg_precision))
        self.logger.info('Accumulated precision: {}'.format(precision))
        self.logger.info('Accumulated Recall: {}'.format(recall))
        log_string = '{} {} {}\n'.format(epoch, avg_precision, recall)
        self.write_log(log_string)

        # generate and save masked images
        evaluator.save_bboxes_masks(epoch, selected_boxes_ind=selected_boxes, mask_draw_precision=0.4, opacity=0.4)
        self.logger.info('Masked images + bboxes have been saved to {}'.format(self.save_dir_test))

    @torch.no_grad()
    def test(self, num_masks=5):
        """
        Runs a test on a trained model and saves a selected number of generated masks
        Parameters
        ----------
        num_masks
        """
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

    def write_log(self, msg):
        addr = os.path.join(self.save_dir, self.log_name)
        with open(addr, 'a') as file:
            file.write(msg)
