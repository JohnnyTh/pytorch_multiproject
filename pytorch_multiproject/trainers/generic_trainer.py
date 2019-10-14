import logging
import torch
from trainers.base_trainer import BaseTrainer

class GenericTrainer(BaseTrainer):

    def __init__(self, model, criterion, optimizer, scheduler, metrics, epochs, checkpoint):
        self.logger = logging.getLogger('trainer')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_metrics = metrics
        self.epochs = epochs
        self.start_epoch = 1
        if checkpoint is not None:
            self._deserialize(checkpoint)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            res = self._train_step(self.model)
            if res['best_performance']:
                self._serialize(epoch)

    def _serialize(self, epoch):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer': {'name': self.optimizer.__class__.__name__,
                          'state': self.optimizer.state_dict()},
            'best_metrics': self.best_metrics
        }
        file_path = 'xxx.pth'
        torch.save(state, file_path)

    def _deserialize(self, load_path):
        checkpoint = torch.load(load_path)
        self.start_epoch = checkpoint['epoch']+1
        self.model.load_state_dict(checkpoint['model_state'])
        self.best_metrics = checkpoint['best_metrics']
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['optimizer']['name'] != self.optimizer.__class__.__name__:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                 "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer']['state'])
