import logging
import os
from utils import weights_init
from abc import abstractmethod


class BaseTrainer:

    def __init__(self, model, criterion, optimizer, config):
        self.ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                            filename=os.path.join(self.ROOT_DIR, 'saved', 'mnist', 'log', 'info.log'))
        self.logger = logging.getLevelName('train')
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

    @abstractmethod
    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        pass

    def initialize_weights(self):
        self.model.apply(weights_init)
        self.logger.info('ROOT_DIR: {}'.format(self.ROOT_DIR))
        self.logger.info('Model weights initialized!')
