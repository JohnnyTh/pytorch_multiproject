import os
import logging
import torch
from trainers.generic_trainer import GenericTrainer
from sklearn.metrics import classification_report


class AgeGenderTrainer(GenericTrainer):

    def __init__(self,  dataloaders, scheduler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Trainer implementing single training step behavior for AgeGenderModel.
            Args:
                *args: root, model, criterion, optimizer, metrics, epochs
                **kwargs: checkpoint (default=None)
                dataloaders (dict): a dict containing 'train' and 'val' dataloaders
                scheduler (lr_scheduler): learning rate scheduler
                
                Note: best_metrics = { 'loss': {'gender' : 10.0, 'age': 100.0, 'total' : 100.0},
                                       'acc_gender' : 0.0}
        """
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.logger = logging.getLogger(os.path.basename(__file__))

    def _train_step(self, epoch):
        self.logger.info('\n\n' +'Epoch {}/{}'.format(epoch, self.epochs))
        self.logger.info('-' * 10)
        results = {
            'best_performance': False
        }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()   # Set model to evaluate mode

            running_metrics = {
                'loss': {'gender': 0.0, 'age': 0.0, 'total': 0.0},
                'acc_gender': 0.0
            }

            # Collect y_hat, y_true values for gender classification report
            y_hat = torch.Tensor([])
            y_true = torch.Tensor([])

            for inputs, labels_gender, labels_age in self.dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels_gender = labels_gender.to(self.device)
                labels_age = labels_age.to(self.device)
                labels_gender = labels_gender.view(-1, 1)
                labels_age = labels_age.view(-1, 1)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_gender, outputs_age = self.model(inputs)
                    # Round the outputs of sigmoid func to obtain predicted class

                    loss_gender = self.criterion['gender'](outputs_gender, labels_gender)
                    loss_age = self.criterion['age'](outputs_age, labels_age)
                    # Total loss is calculated as a sum of two losses for gender and
                    # age multiplied by respective correction coefficients
                    loss = 1. * loss_gender + 0.25 * loss_age

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # Statistics collection
                preds = outputs_gender.round()
                y_hat = torch.cat((y_hat, preds.cpu()))
                y_true = torch.cat((y_true, labels_gender.cpu()))

                running_metrics['loss']['gender'] += loss_gender.item() * inputs.size(0)
                running_metrics['loss']['age'] += loss_age.item() * inputs.size(0)
                running_metrics['loss']['total'] += loss.item() * inputs.size(0)
                running_metrics['acc_gender'] += torch.sum(preds==labels_gender.data)
            if phase == 'train' and self.scheduler is not None:
                self.scheduler.step()

            epoch_metrics = {'loss': None, 'acc_gender': None}
            epoch_metrics['loss'] = {key: running_metrics['loss'][key] / len(self.dataloaders[phase].dataset)
                                     for key in running_metrics['loss'].keys()}
            epoch_metrics['acc_gender'] = running_metrics['acc_gender'].double() / len(self.dataloaders[phase].dataset)

            # Output epoch results
            self.logger.info('>>> {} phase <<<'.format(phase))
            self.logger.info('Gender: loss: {:.4f} acc: {:.4f}'.format(epoch_metrics['loss']['gender'],
                                                             epoch_metrics['acc_gender']))
            self.logger.info('Age: loss (MAE): {:.4f}'.format(epoch_metrics['loss']['age']))
            self.logger.info('Total Loss: {}'.format(epoch_metrics['loss']['total']))
            self.logger.info('')

            if epoch % 1 == 0:
                self.logger.info('\n' + '         ---- Gender classification report: ----' +
                                 '\n' + classification_report(y_true.detach(), y_hat.detach(), target_names=['Female', 'Male']))

            # Check if we got the best performance based on the selected criteria
            if (
                phase == 'val'
                and epoch_metrics['loss']['total'] < self.best_metrics['loss']['total']
                and epoch_metrics['acc_gender'] > self.best_metrics['acc_gender']
                and epoch_metrics['loss']['age'] < self.best_metrics['loss']['age']
               ):
                self.best_metrics = epoch_metrics
                results['best_performance'] = True

        return results
