import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import torch
from mock import patch
from trainers.mnist_trainer import MnistTrainer


class ModelTestMock:

    def __init__(self):
        self.training = True
        self.is_cuda = False

    def __call__(self, *args, **kwargs):
        return torch.randn((64, 10))

    def __next__(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return self

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def state_dict(self):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass


class OptTestMock:

    def zero_grad(self):
        pass

    def step(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass


class LrSchedMock:

    def step(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass


class DataloaderTestMock:

    def __init__(self, n):
        self.n = n
        self.num = 0
        self.dataset = torch.randn(640)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            cur, self.num = self.num, self.num + 1
            return torch.randn((64, 1, 28, 28)), torch.randint(10, (64, 1))
        else:
            self.num = 0
            raise StopIteration()

    def __len__(self):
        return self.n


class LossTestMock:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return torch.randn(1)


test_data = {
    'dataloaders': {'train': DataloaderTestMock(10),
                    'val': DataloaderTestMock(10)},
    'root': '/home',
    'model': ModelTestMock(),
    'criterion': LossTestMock(),
    'optimizer': OptTestMock(),
    'scheduler': LrSchedMock(),
    'metrics': {'epoch': 0, 'loss': 10., 'acc': 0.0},
    'epochs': 10,
    'checkpoint': {'start_epoch': 6}
}

deserialize_data = {
    'epoch': 5,
    'model_state': 'chkpt_model',
    'best_metrics': {'epoch': 0, 'loss': 10., 'acc': 0.0},
    'optimizer': {
        'name': OptTestMock().__class__.__name__,
        'state': 'chkpt_optim'},
    'scheduler': {
        'name': LrSchedMock().__class__.__name__,
        'state': 'chkpt_sched'
    }
}


@patch('os.mkdir')
@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.mnist_trainer.MnistTrainer._serialize', return_value=None)
def test_train_run(self, _, __):
    trainer = MnistTrainer(test_data['dataloaders'], root=test_data['root'], model=test_data['model'],
                           criterion=test_data['criterion'], optimizer=test_data['optimizer'],
                           scheduler=test_data['scheduler'], metrics=test_data['metrics'], epochs=test_data['epochs'])
    trainer.train()


@patch('os.mkdir')
@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.mnist_trainer.MnistTrainer._serialize', return_value=None)
@patch('torch.load', return_value=deserialize_data)
def test_train_deserialize_and_run(self, _, __, ___):
    # Assuming we trained the model from epoch 1 to 5, then saved it and now want to restart
    trainer = MnistTrainer(test_data['dataloaders'], root=test_data['root'], model=test_data['model'],
                           criterion=test_data['criterion'], optimizer=test_data['optimizer'],
                           scheduler=test_data['scheduler'], metrics=test_data['metrics'],
                           epochs=test_data['epochs'])

    trainer._deserialize('/does_not_matter')
    assert trainer.start_epoch == 6
    assert trainer.epochs == test_data['epochs'] + 6
    trainer.train()
