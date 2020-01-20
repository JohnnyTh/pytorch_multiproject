import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.insert(0, ROOT_DIR)
import logging
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.word2vec import Word2VecModel
from data.word2vec_dataset import Word2VecDataset
from trainers.word2vec_trainer import Word2VecTrainer
from logger.logger import main_run, default_log_config
import data.custom_transforms as t_custom

# default configuration file with hyperparameters
DEFAULT_CONFIG = 'train.json'


def main(config, args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create an instance of logger
    logger = logging.getLogger(os.path.basename(__file__))
    if args.resource_dir is not None:
        resources_dir = args.resource_dir
    else:
        resources_dir = os.path.join(ROOT_DIR, 'resources', config.get('resource_dir', 'cornell movie-dialogs corpus'))

    data = os.path.join(resources_dir, 'data')
    vocabulary = pickle.load(os.path.join(os.path.join(resources_dir, 'vocabulary'), 'vocabulary.pickle'))
    word2idx = pickle.load(os.path.join(os.path.join(resources_dir, 'word2idx'), 'word2idx.pickle'))
    idx2word = pickle.load(os.path.join(os.path.join(resources_dir, 'idx2word'), 'idx2word.pickle'))
    word_count = pickle.load(os.path.join(os.path.join(resources_dir, 'word_counts'), 'word_counts.pickle'))
    word_count = dict(sorted(word_count.items()))

    word_freq = np.array([word_count[idx] for idx in word_count])
    word_freq = word_freq / word_freq.sum()

    dataset = Word2VecDataset(resources_dir, word2idx, idx2word, word_freq=None, data_paths=[data], extensions=(('.pickle'),))
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    model = Word2VecModel(vocab_size=len(vocabulary), word_freq=word_freq)
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=config.get('lr', 0.0002))

    # define number of epochs
    epochs = config.get('epochs', 100)

    trainer = Word2VecTrainer(dataloaders=data_loader, root=ROOT_DIR, model=model, criterion=None,
                              optimizer=optim, epochs=epochs, save_dir=args.save_dir, checkpoint=args.checkpoint)

    trainer.train()


if __name__ == '__main__':
    default_log_config()
    main_run(main, DEFAULT_CONFIG)
