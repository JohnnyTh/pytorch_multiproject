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

    def eval(self):
        self.training = False

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
    'dataloaders': {'train': DataloaderTestMock(50),
                    'val': DataloaderTestMock(50)},
    'root': '/home',
    'model': ModelTestMock(),
    'criterion': None,
    'optimizer': MagicMock(),
    'scheduler': MagicMock(),
    'metrics': {'loss_gen': 100.0, 'ab_disc_loss': 100.0, 'ba_disc_loss': 100.0},
    'epochs': 10,
    'checkpoint': {'start_epoch': 6}

}

deserialize_data = {
    'epoch': 5,
    'model_state': 'chkpt_model',
    'best_metrics': {'loss_gen': 100.0, 'ab_disc_loss': 100.0, 'ba_disc_loss': 100.0},
    'optimizer': {
        'name': MagicMock().__class__.__name__,
        'state': 'chkpt_optim'},
    'scheduler': {
        'name': MagicMock().__class__.__name__,
        'state': 'chkpt_sched'
    }
}


@patch('os.mkdir')
@patch('trainers.cycle_gan_trainer.save_image')
@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.cycle_gan_trainer.CycleGanTrainer._serialize', return_value=None)
def test_train_run(self, _, __, ___):
    trainer = CycleGanTrainer(dataloaders=test_data['dataloaders'], root=test_data['root'],
                              model=test_data['model'], criterion=test_data['criterion'],
                              optimizer=test_data['optimizer'], scheduler=test_data['scheduler'],
                              metrics=test_data['metrics'], epochs=test_data['epochs'])
    trainer.train()


@patch('os.mkdir')
@patch('trainers.cycle_gan_trainer.save_image')
@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.cycle_gan_trainer.CycleGanTrainer._serialize', return_value=None)
@patch('torch.load', return_value=deserialize_data)
def test_train_deserialize_and_run(self, _, __, ___, ____):
    # Assuming we trained the model from epoch 1 to 5, then saved it and now want to restart
    trainer = CycleGanTrainer(dataloaders=test_data['dataloaders'], root=test_data['root'],
                              model=test_data['model'], criterion=test_data['criterion'],
                              optimizer=test_data['optimizer'], scheduler=test_data['scheduler'],
                              metrics=test_data['metrics'], epochs=test_data['epochs'])
    trainer._deserialize('/does_not_matter')
    assert trainer.start_epoch == 6
    assert trainer.epochs == test_data['epochs'] + 6
    trainer.train()


@patch('os.mkdir')
@patch('trainers.cycle_gan_trainer.save_image')
@patch('torch.load', return_value=deserialize_data)
def test_test_run(self, _, __):
    trainer = CycleGanTrainer(dataloaders=test_data['dataloaders']['val'], root=test_data['root'],
                              model=test_data['model'], criterion=test_data['criterion'],
                              optimizer=test_data['optimizer'], scheduler=test_data['scheduler'],
                              metrics=test_data['metrics'], epochs=test_data['epochs'])
    trainer.test()
