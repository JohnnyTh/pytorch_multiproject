import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import torch
from mock import MagicMock
from mock import patch
from trainers.cycle_gan_trainer import CycleGanTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelTestMock:

    def __init__(self):
        self.training = True
        self.is_cuda = False

    def __call__(self, real_a, real_b, step_flag,
                fake_b_disc=None, fake_a_disc=None):
        if step_flag == 'gen_step':
            return torch.rand((1, 3, 32, 32)), torch.rand((1, 3, 32, 32)), \
                   torch.rand((1, 3, 32, 32)), torch.rand((1, 3, 32, 32)), torch.rand(1)*5
        elif step_flag == 'disc_step':
            return torch.rand(1)*5, torch.rand(1)*5

    def __next__(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return self

    def train(self):
        self.training = True

    def state_dict(self):
        pass

    def load_state_dict(self, *args, **kwargs):
        pass


class DataloaderTestMock:

    def __init__(self, n):
        self.n = n
        self.num = 0
        self.dataset = torch.randn(100)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # first out - source img, second - target img
        if self.num < self.n:
            cur, self.num = self.num, self.num + 1
            return torch.rand((1, 3, 256, 256)), torch.rand((1, 3, 256, 256))
        else:
            self.num = 0
            raise StopIteration()

    def __len__(self):
        return self.n


test_data = {
    'dataloader': DataloaderTestMock(100),
    'root': '/home',
    'model': ModelTestMock(),
    'criterion': None,
    'optimizer': MagicMock(),
    'scheduler': MagicMock(),
    'metrics': {'loss_gen': 100.0, 'ab_disc_loss': 100.0, 'ba_disc_loss': 100.0},
    'epochs': 10,
    'checkpoint': {'start_epoch': 6}

}


@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.cycle_gan_trainer.CycleGanTrainer._serialize', return_value=None)
def test_train_run(self, _):
    trainer = CycleGanTrainer(dataloader=test_data['dataloader'], root=test_data['root'],
                              model=test_data['model'], criterion=test_data['criterion'],
                              optimizer=test_data['optimizer'], scheduler=test_data['scheduler'],
                              metrics=test_data['metrics'], epochs=test_data['epochs'])
    trainer.train()