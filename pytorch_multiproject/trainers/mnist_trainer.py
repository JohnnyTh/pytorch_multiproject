from .base_trainer import BaseTrainer


class MnistTrainer(BaseTrainer):

    def __init__(self, model, criterion, optimizer,
                 config, data_loader, val_data_loader,
                 scheduler):
        super().__init__(model, criterion, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.scheduler = scheduler


    def train_epoch(self, epoch):
        pass
