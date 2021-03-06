import os
import logging
import torch
from trainers.base_trainer import BaseTrainer
from abc import abstractmethod


class GenericTrainer(BaseTrainer):

    def __init__(self, root, model, criterion, optimizer, scheduler, metrics, epochs, hyperparams=None,
                 save_dir=None, checkpoint=None, change_lr=False):
        """ Generic trainer; implements train(), _serialize(), and _deserialize methods.
            root (str): project root directory.
            model (callable): an instance of custom neural network class inheriting from nn.Module class.
            criterion (callable): a loss function.
            optimizer (optimizer object): object implementing optimization algorithm.
            scheduler (lr_scheduler): learning rate scheduler object, changes lr of the optimizer every time step()
                                      method is called.
            metrics (dict): dict containing metrics, specific for every custom trainer implementation
            epochs (int): number of training epochs
            hyperparams (dict): various hyperparameters we might need inside GenericTrainer
                                or its custom implementations
            save_dir (str): dir to save the trained models and generated images
            checkpoint (str, optional): checkpoint path to resume the training from
            change_lr (bool): if True, learning rate of the optimizer will be changed to provided value even
                              if the model is restored from checkpoint. Uses self.hyperparams['lr'] value.
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
        if hyperparams is None:
            hyperparams = dict({})
        self.hyperparams = hyperparams
        self.start_epoch = 1
        self.generic_logger = logging.getLogger(os.path.basename(__file__))
        self.change_lr = change_lr

        if save_dir is not None:
            self.save_dir = save_dir
        else:
            # if custom save dir not provided, save in project folder instead
            self.save_dir = os.path.join(self.root, 'saved')
        # create a directory for saving the output
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

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
        if self.scheduler is not None:
            sched_state = {'name': self.scheduler.__class__.__name__,
                           'state': self.scheduler.state_dict()}
        else:
            sched_state = None

        state = {
            'epoch': epoch,
            'model_name': self.name,
            'model_state': self.model.state_dict(),
            'optimizer': {'name': self.optimizer.__class__.__name__,
                          'state': self.optimizer.state_dict()},
            'scheduler': sched_state,
            'best_metrics': self.best_metrics
        }
        chkpt = '{}_epoch_{}.pth'.format(self.name, epoch)
        file_path = os.path.join(self.save_dir, chkpt)
        torch.save(state, file_path)
        self.generic_logger.info('Saving the model at {}'.format(file_path))

    def _deserialize(self, load_path):
        # restore the model and other parameters from the checkpoint file ('xxx.pth')
        checkpoint = torch.load(load_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.epochs = self.epochs + self.start_epoch
        self.model.load_state_dict(checkpoint['model_state'])
        self.best_metrics = checkpoint['best_metrics']

        # restore optimizer from checkpoint
        if checkpoint['optimizer']['name'] != self.optimizer.__class__.__name__:
            self.logger.warning("Warning: Given optimizer type  is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer']['state'])

        # manually adjust the lr of the optimizer
        if self.change_lr is True:
            try:
                # standard pytorch optimizer is an iterable
                iter(self.optimizer)
                for param_group in self.optimizer:
                    param_group['lr'] = self.hyperparams.get('lr', 0.0002)
            # if we are using custom optimizer, it is not iterable
            except TypeError:
                if hasattr(self.optimizer, 'change_lr'):
                    self.optimizer.change_lr(self.hyperparams.get('lr', 0.0002))
                    self.logger.info('Learning rate has been changed!')
                else:
                    raise NotImplementedError('required method change_lr not implemented in provided optimizer object')

        # restore scheduler parameters from the checkpoint
        if checkpoint['scheduler'] is not None:
            if checkpoint['scheduler']['name'] != self.scheduler.__class__.__name__:
                self.logger.warning("Warning: Given scheduler type  is different from that of checkpoint. "
                                    "Scheduler parameters not being resumed.")
            else:
                self.scheduler.load_state_dict(checkpoint['scheduler']['state'])

        self.logger.info('Resuming from checkpoint...')
