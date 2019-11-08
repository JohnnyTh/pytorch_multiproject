import os
import logging
import torch
from trainers.generic_trainer import GenericTrainer
from torchvision.utils import save_image
from tqdm import tqdm


class CycleGanTrainer(GenericTrainer):

    def __init__(self, dataloaders, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Trainer implementing single training step behavior for AgeGenderModel.
            Args:
                *args: root, model, criterion, optimizer, metrics, epochs
                **kwargs: checkpoint (default=None)
                dataloader ():  DESCRIPTION HERE 
                scheduler (lr_scheduler): learning rate scheduler
        """
        self.dataloaders = dataloaders
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.save_dir_test = os.path.join(self.save_dir, 'gan_test')
        if not os.path.exists(self.save_dir_test):
            os.mkdir(self.save_dir_test)

    def _train_step(self, epoch):
        self.logger.info('Epoch {}/{}'.format(epoch, self.epochs))
        self.logger.info('-' * 10)

        # print parameters of optimizer and scheduler every epoch
        self.logger.info(str(self.optimizer))
        if self.scheduler is not None:
            self.logger.info(str(self.scheduler))

        results = {
            'best_performance': False
        }

        running_metrics = {
            'loss_gen': 0.0,
            'ab_disc_loss': 0.0,
            'ba_disc_loss': 0.0
        }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()   # Set model to evaluate mode

            t = tqdm(iter(self.dataloaders[phase]), leave=False, total=len(self.dataloaders[phase]))
            for idx, images in enumerate(t):

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                img_source = images[0].to(device)
                img_target = images[1].to(device)
                # forward pass trough generators
                fake_b, fake_a, rec_a, rec_b, loss_gen = self.model(img_source, img_target, 'gen_step')

                # backpropagation, optimization, etc. are done only in the train phase
                if phase == 'train':
                    self.optimizer.zero_grad('optim_gen')
                    loss_gen.backward()
                    self.optimizer.step('optim_gen')

                    # get losses from discriminators
                    ab_disc_loss, ba_disc_loss = self.model(img_source, img_target, 'disc_step',
                                                            fake_b_disc=fake_b, fake_a_disc=fake_a)

                    self.optimizer.zero_grad('optim_disc')
                    ab_disc_loss.backward()
                    ba_disc_loss.backward()
                    self.optimizer.step('optim_disc')

                    running_metrics['loss_gen'] += loss_gen.item() * img_source.size(0)
                    running_metrics['ab_disc_loss'] += ab_disc_loss.item() * img_source.size(0)
                    running_metrics['ba_disc_loss'] += ba_disc_loss.item() * img_source.size(0)

                elif phase == 'val':
                    # save the generated image files
                    images = {'real_a': img_source, 'real_b': img_target,
                              'fake_b': fake_b, 'fake_a': fake_a,
                              'rec_a': rec_a, 'rec_b': rec_b}
                    self._save_img(images, idx)
                    if idx == len(self.dataloaders[phase]) - 1:
                        self.logger.info('The transformed images have been saved to {}'.format(self.save_dir_test))

            if phase == 'train':

                if self.scheduler is not None:
                    self.scheduler.step('sched_gen')
                    self.scheduler.step('sched_disc')

                epoch_metrics = {key: running_metrics[key]/len(self.dataloaders[phase].dataset)
                                 for key in running_metrics.keys()}

                # Output epoch results
                self.logger.info('Generators loss: {:.4f}'.format(epoch_metrics['loss_gen']))
                self.logger.info('AB discriminator loss: {:.4f}'.format(epoch_metrics['ab_disc_loss']))
                self.logger.info('BA discriminator loss: {:.4f}'.format(epoch_metrics['ba_disc_loss']))

                # always save the model after each iteration
                self.best_metrics = epoch_metrics
                results['best_performance'] = True


        return results

    def test(self):
        self.logger.info('Running test. The model was restored from epoch {}'.format(self.start_epoch - 1))
        self.model.eval()
        t = tqdm(iter(self.dataloaders), leave=False, total=len(self.dataloaders))
        for idx, images in enumerate(t):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img_source = images[0].to(device)
            img_target = images[1].to(device)
            # forward pass trough generators
            fake_b, fake_a, rec_a, rec_b, loss_gen = self.model(img_source, img_target, 'gen_step')

            # save the generated image files
            images = {'real_a': img_source, 'real_b': img_target,
                      'fake_b': fake_b, 'fake_a': fake_a,
                      'rec_a': rec_a, 'rec_b': rec_b}
            self._save_img(images, idx)
        self.logger.info('The transformed images have been saved to {}'.format(self.save_dir_test))

    def _save_img(self, images, idx):
        for key in images.keys():
            save_image(images[key], os.path.join(self.save_dir_test, key+'_{}.png'.format(idx)))
