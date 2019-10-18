from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def _train_step(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def _serialize(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _deserialize(self, chckpt_path):
        raise NotImplementedError
