import os
import torch
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from trainers.generic_trainer import GenericTrainer


class MnistTrainer(GenericTrainer):

    def __init__(self, dataloaders, *args,  **kwargs):
        super().__init__(*args, **kwargs)
        """Trainer implementing single training step behavior for MnistNet.
            Args:
                *args: root, model, criterion, optimizer, metrics, epochs
                **kwargs: checkpoint (default=None)
                dataloaders (dict): a dict containing 'train' and 'val' dataloaders
                scheduler (lr_scheduler): learning rate scheduler
        """
        self.dataloaders = dataloaders
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
                self.model.eval()  # Set model to evaluate mode

            running_metrics = {
                'loss': 0.0,
                'acc': 0.0
            }
            # Collect y_hat, y_true values for classification report
            y_hat = torch.Tensor([])
            y_true = torch.Tensor([])

            # Run the training loop
            for inputs, labels in self.dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # Statistics collection
                preds = outputs.data.argmax(dim=1).cpu().float()
                running_metrics['loss'] += loss.item()*inputs.size(0)
                running_metrics['acc'] += accuracy_score(labels.cpu(), preds)

                y_hat = torch.cat((y_hat, preds))
                y_true = torch.cat((y_true, labels.cpu().float()))

            if phase == 'train' and self.scheduler is not None:
                self.scheduler.step()
            epoch_metrics = {
                'epoch': epoch,
                'loss': None,
                'acc': None
            }
            epoch_metrics['loss'] = running_metrics['loss'] / len(self.dataloaders[phase].dataset)
            # Divide the accumulated accuracy score by the number of minibatches in dataloader
            epoch_metrics['acc'] = running_metrics['acc'] / len(self.dataloaders[phase])
            self.logger.info('>>> {} phase <<<'.format(phase))
            self.logger.info('Loss: {:.4f} Error: {:.4f} %'.format(epoch_metrics['loss'], (1 - epoch_metrics['acc'])*100))
            self.logger.info(' ')

            if epoch % 1 == 0:
                self.logger.info('\n' + '         ---- Classification report: ----' +
                                 '\n' + classification_report(y_true, y_hat))
            if (
                phase == 'val'
                and epoch_metrics['loss'] < self.best_metrics['loss']
                and epoch_metrics['acc'] > self.best_metrics['acc']
               ):
                self.best_metrics = epoch_metrics
                self.logger.info('Best model performance so far at epoch {}!'.format(epoch))
                results['best_performance'] = True

        return results

