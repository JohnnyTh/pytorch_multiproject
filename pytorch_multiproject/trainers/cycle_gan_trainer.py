import os
import logging
import torch
from trainers.generic_trainer import GenericTrainer


class CycleGanTrainer(GenericTrainer):

    def __init__(self, dataloader, scheduler=None, save_chkpt=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Trainer implementing single training step behavior for AgeGenderModel.
            Args:
                *args: root, model, criterion, optimizer, metrics, epochs
                **kwargs: checkpoint (default=None)
                dataloader ():  DESCRIPTION HERE 
                scheduler (lr_scheduler): learning rate scheduler
        """
        self.save_chckpt = save_chkpt
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.logger = logging.getLogger(os.path.basename(__file__))

    def _train_step(self, epoch):
        self.logger.info('Epoch {}/{}'.format(epoch, self.epochs))
        self.logger.info('-' * 10)
        results = {
            'best_performance': False
        }

        running_metrics = {
            'loss_gen': 0.0,
            'ab_disc_loss': 0.0,
            'ba_disc_loss': 0.0
        }

        for img_source, img_target in self.dataloader:

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img_source = img_source.to(device)
            img_target = img_target.to(device)
            self.model.train()
            # forward pass trough generators
            fake_b, fake_a, rec_a, rec_b, loss_gen = self.model(img_source, img_target, 'gen_step')

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

        if self.scheduler is not None:
            self.scheduler.step('sched_gen')
            self.scheduler.step('sched_disc')

        epoch_metrics = {key: running_metrics[key]/len(self.dataloader.dataset)
                         for key in running_metrics.keys()}

        # Output epoch results
        self.logger.info('Generators loss: {:.4f}'.format(epoch_metrics['loss_gen']))
        self.logger.info('AB discriminator loss: {:.4f}'.format(epoch_metrics['ab_disc_loss']))
        self.logger.info('BA discriminator loss: {:.4f}'.format(epoch_metrics['ba_disc_loss']))

        if (
            epoch_metrics['loss_gen'] < self.best_metrics['loss_gen']
            and epoch_metrics['ab_disc_loss'] < self.best_metrics['ab_disc_loss']
            and epoch_metrics['ba_disc_loss'] < self.best_metrics['ba_disc_loss']
        ):
            self.best_metrics = epoch_metrics
            if self.save_chckpt:
                results['best_performance'] = True

        return results

