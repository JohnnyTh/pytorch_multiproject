import os
import logging
import torch
from trainers.base_trainer import BaseTrainer
from abc import abstractmethod


class GenericTrainer(BaseTrainer):

    def __init__(self, root, model, criterion, optimizer, scheduler, metrics, epochs, checkpoint=None):
        """ Generic trainer implements train(), _serialize(), and _deserialize methods.
            root (str): project root directory
            model (callable): an instance of custom neural network class inheriting from nn.Module class.
            criterion (callable): a loss function.
            optimizer (optimizer object): object implementing optimization algorithm
            metrics (dict): dict containing metrics, specific for every custom trainer implementation
            epochs (int): number of training epochs
            checkpoint (str, optional): checkpoint path to resume training from
        """
        self.logger = logging.getLogger('trainer')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root = root
        self.model = model.to(self.device)
        self.name = model.__class__.__name__
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_metrics = metrics
        self.epochs = epochs
        self.start_epoch = 1
        self.generic_logger = logging.getLogger(os.path.basename(__file__))
        if checkpoint is not None:
            self._deserialize(checkpoint)

    @abstractmethod
    def _train_step(self, epoch):
        # this method is implemented in custom Trainers
        raise NotImplementedError

    def train(self):
        # training loop through epochs, saves the model if some criteria are satisfied during the training
        for epoch in range(self.start_epoch, self.epochs+1):
            res = self._train_step(epoch)
            if res['best_performance']:
                self._serialize(epoch)

    def _serialize(self, epoch):
        # save the model and some other parameters

        state = {
            'epoch': epoch,
            'model_name': self.name,
            'model_state': self.model.state_dict(),
            'optimizer': {'name': self.optimizer.__class__.__name__,
                          'state': self.optimizer.state_dict()},
            'scheduler': {'name': self.scheduler.__class__.__name__,
                          'state': self.scheduler.state_dict()},
            'best_metrics': self.best_metrics
        }
        chkpt = '{}_best.pth'.format(self.name)
        file_path = os.path.join(self.root, 'saved', chkpt)
        torch.save(state, file_path)
        self.generic_logger.info('Saving the model at {}'.format(file_path))

    def _deserialize(self, load_path):
        # restore the model and other parameters from the checkpoint file ('xxx.pth')
        checkpoint = torch.load(load_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.epochs = self.epochs + self.start_epoch + 1
        self.model.load_state_dict(checkpoint['model_state'])
        self.best_metrics = checkpoint['best_metrics']
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['optimizer']['name'] != self.optimizer.__class__.__name__:
            self.logger.warning("Warning: Given optimizer type  is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer']['state'])
        if checkpoint['scheduler']['name'] != self.scheduler.__class__.__name__:
            self.logger.warning("Warning: Given scheduler type  is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.scheduler.load_state_dict(checkpoint['scheduler']['state'])
