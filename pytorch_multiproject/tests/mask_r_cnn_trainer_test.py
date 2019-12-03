import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import torch
from mock import MagicMock
from mock import patch
from trainers.mask_r_cnn_trainer import MaskRCNNTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelTestMock:

    def __init__(self):
        self.training = True
        self.is_cuda = False

    def __call__(self, images, *args):
        if self.training:
            losses_dict = {'loss_box_reg': torch.rand(1)*10,
                           'loss_classifier': torch.rand(1)*10,
                           'loss_mask': torch.rand(1)*10,
                           'loss_objectness': torch.rand(1)*10,
                           'loss_rpn_box_reg': torch.rand(1)*10}
            return losses_dict
        else:
            dict_img_1 = {'boxes': torch.rand((100, 4)) * torch.randint(450, ((1), )),
                          'labels': torch.ones(100),
                          'masks': torch.randint(2, (100, 1, 450, 450)),
                          'scores': torch.FloatTensor(100).uniform_(0.6, 0.95)}

            dict_img_2 = {'boxes': torch.rand((100, 4)) * torch.randint(450, ((1),)),
                          'labels': torch.ones(100),
                          'masks': torch.randint(2, (100, 1, 450, 450)),
                          'scores': torch.FloatTensor(100).uniform_(0.6, 0.95)}
            return [dict_img_1, dict_img_2]

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

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # first out - source img, second - target img
        if self.num < self.n:
            cur, self.num = self.num, self.num + 1

            img = (torch.rand(3, 450, 450), )
            target_dict = {'area': torch.rand(2)*1000,
                           'boxes': torch.randint(450, (2, 4)),
                           'image_id': torch.randint(1000, ((1), )),
                           'iscrowd': torch.zeros(2),
                           'labels': torch.ones(2),
                           'masks': torch.randint(2, (2, 450, 450))}
            target = (target_dict, )

            return tuple([img, target])
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
    'metrics': {},
    'epochs': 10,
    'checkpoint': {'start_epoch': 6}

}

deserialize_data = {
    'epoch': 5,
    'model_state': 'chkpt_model',
    'best_metrics': {},
    'optimizer': {
        'name': MagicMock().__class__.__name__,
        'state': 'chkpt_optim'},
    'scheduler': {
        'name': MagicMock().__class__.__name__,
        'state': 'chkpt_sched'
    }
}


@patch('os.mkdir')
@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.cycle_gan_trainer.CycleGanTrainer._serialize', return_value=None)
def test_train_run(self, _, __):
    trainer = MaskRCNNTrainer(dataloaders=test_data['dataloaders'], root=test_data['root'],
                              model=test_data['model'], criterion=test_data['criterion'],
                              optimizer=test_data['optimizer'], scheduler=test_data['scheduler'],
                              metrics=test_data['metrics'], epochs=test_data['epochs'])
    trainer.train()


"""
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

"""