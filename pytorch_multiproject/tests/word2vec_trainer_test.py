import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)

import torch
import numpy as np
from mock import MagicMock
from mock import patch
from trainers.word2vec_trainer import Word2VecTrainer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


vocab_size = 100
emb_size = 10
embeddings = np.random.rand(vocab_size, emb_size)


class ModelTestMock:

    def __init__(self):
        self.training = True
        self.is_cuda = False
        input_embeddings = MagicMock()
        input_embeddings.weight.data.cpu().numpy().return_value = embeddings
        self.input_embeddings = input_embeddings

    def __call__(self, input_word, target_words):
        loss = torch.randn(1) * 10
        return loss

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


class DatasetMock:

    def __init__(self, size=32*5):
        self.vocab_size = vocab_size
        self.window_size = 5

        input_word = torch.randint(self.vocab_size, size=(size, 1)).tolist()
        target_words = torch.randint(self.vocab_size, size=(size, self.window_size*2)).tolist()
        self.data = [[input_w, target_w] for input_w, target_w in zip(input_word, target_words)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_word, target_words = self.data[item]
        return torch.tensor(input_word).long(), torch.tensor(target_words).long()

    def subsample_or_get_data(self):
        pass


test_dataset = DatasetMock()
test_data = {
    'dataloader_params': {'dataset': test_dataset, 'batch_size': 32, 'shuffle': True,
                          'num_workers': 0},
    'dataset': DatasetMock(),
    'root': '/home',
    'model': ModelTestMock(),
    'optimizer': MagicMock(),
    'epochs': 10,
    'checkpoint': {'start_epoch': 6}
}

deserialize_data = {
    'epoch': 5,
    'model_state': 'chkpt_model',
    'best_metrics': None,
    'optimizer': {
        'name': MagicMock().__class__.__name__,
        'state': 'chkpt_optim'},
    'scheduler': None
}


@patch('os.mkdir')
@patch('trainers.word2vec_trainer.pickle')
@patch('torch.Tensor.backward', return_value=None)
@patch('trainers.word2vec_trainer.Word2VecTrainer._serialize', return_value=None)
def test_train_run(self, _, __, ___):
    trainer = Word2VecTrainer(dataloader_params=test_data['dataloader_params'], subsample_words=False,
                              root=test_data['root'], model=test_data['model'], criterion=None,
                              optimizer=test_data['optimizer'], scheduler=None, metrics=None, epochs=test_data['epochs'])
    trainer.train()


@patch('os.mkdir')
@patch('trainers.word2vec_trainer.pickle')
@patch('torch.Tensor.backward', return_value=None)
@patch('torch.load', return_value=deserialize_data)
@patch('trainers.word2vec_trainer.Word2VecTrainer._serialize', return_value=None)
def test_train_deserialize_and_run(self, _, __, ___, ____):
    # Assuming we trained the model from epoch 1 to 5, then saved it and now want to restart
    trainer = Word2VecTrainer(dataloader_params=test_data['dataloader_params'], subsample_words=False,
                              root=test_data['root'], model=test_data['model'], criterion=None,
                              optimizer=test_data['optimizer'], scheduler=None, metrics=None, epochs=test_data['epochs'])
    trainer._deserialize('/does_not_matter')
    assert trainer.start_epoch == 6
    assert trainer.epochs == test_data['epochs'] + 6
    trainer.train()
