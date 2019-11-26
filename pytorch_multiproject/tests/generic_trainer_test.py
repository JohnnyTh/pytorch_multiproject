import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
from mock import patch
from trainers.generic_trainer import GenericTrainer


class GenericTest(GenericTrainer):

    def _train_step(self, epoch):
        # initialize abstract method for testing purposes
        pass


class MockModel:

    def __init__(self):
        # create a mock model for testing
        self._state_dict = {'state': 'OK'}

    def __repr__(self):
        return 'model_structure'

    def to(self, device):
        return self

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, dict_):
        self._state_dict = dict_


def optim_load_state(self, dict_):
    self.state_dict = dict_


class MockOptim:

    def __init__(self):
        # Create a mock optimizer for testing
        self._state_dict ={'optim_state': 'OK'}

    def __repr__(self):
        return 'test_optimizer'

    def load_state_dict(self, dict_):
        self._state_dict = dict_

    def state_dict(self):
        return self._state_dict


class MockLrSched:

    def __init__(self):
        self._state_dict ={'sched_state': 'OK'}

    def __repr__(self):
        return 'test_scheduler'

    def load_state_dict(self, dict_):
        self._state_dict = dict_

    def state_dict(self):
        return self._state_dict


init_data = {
    'root': '/home',
    'model': MockModel(),
    'criterion': 'test_criterion',
    'optimizer': MockOptim(),
    'scheduler': MockLrSched(),
    'best_metrics': 'test_metrics',
    'epochs': 1,
    'epoch': 10,
    'checkpoint': '/saved'
}

deserialize_data = {
    'epoch': 5,
    'model_state': 'chkpt_model',
    'best_metrics': 'chkpt_metrics',
    'optimizer': {
        'name': 'MockOptim',
        'state': 'chkpt_optim'},
    'scheduler': {
        'name': 'MockLrSched',
        'state': 'chkpt_sched'
    }
}


@patch('os.mkdir')
def test_initialization(self):
    test_trainer = GenericTest(root=init_data['root'], model=init_data['model'], criterion=init_data['criterion'],
                               optimizer=init_data['optimizer'], scheduler=init_data['scheduler'],
                               metrics=init_data['best_metrics'], epochs=init_data['epochs'])
    assert test_trainer.root == init_data['root']
    assert str(test_trainer.model) == 'model_structure'
    assert test_trainer.name == 'MockModel'
    assert test_trainer.criterion == init_data['criterion']
    assert str(test_trainer.optimizer) == 'test_optimizer'
    assert test_trainer.best_metrics == init_data['best_metrics']
    assert test_trainer.epochs == init_data['epochs']


def side_effect_serialize(state, file_path):
    chkpt = '{}_epoch_{}.pth'.format('MockModel', init_data['epoch'])
    test_path = os.path.join(init_data['root'], 'saved', chkpt)
    assert state['epoch'] == init_data['epoch']
    assert state['model_name'] == 'MockModel'
    assert state['model_state'] == {'state': 'OK'}
    assert state['optimizer']['name'] == 'MockOptim'
    assert state['optimizer']['state'] == {'optim_state': 'OK'}
    assert state['best_metrics'] == init_data['best_metrics']
    assert file_path == test_path


@patch('os.mkdir')
@patch('torch.save', side_effect=side_effect_serialize)
def test_serialize(self, _):
    test_trainer = GenericTest(root=init_data['root'], model=init_data['model'], criterion=init_data['criterion'],
                               optimizer=init_data['optimizer'], scheduler=init_data['scheduler'],
                               metrics=init_data['best_metrics'], epochs=init_data['epochs'])

    test_trainer._serialize(init_data['epoch'])


@patch('os.mkdir')
@patch('torch.load', return_value=deserialize_data)
def test_deserialize(self, _):
    test_trainer = GenericTest(root=init_data['root'], model=init_data['model'], criterion=init_data['criterion'],
                               optimizer=init_data['optimizer'], scheduler=init_data['scheduler'],
                               metrics=init_data['best_metrics'], epochs=init_data['epochs'])
    test_trainer._deserialize('/does_not_matter')
    assert test_trainer.start_epoch == deserialize_data['epoch'] + 1
    assert test_trainer.epochs == init_data['epochs'] + test_trainer.start_epoch
    assert test_trainer.model.state_dict() == deserialize_data['model_state']
    assert test_trainer.best_metrics == deserialize_data['best_metrics']
    assert test_trainer.optimizer.state_dict() == deserialize_data['optimizer']['state']