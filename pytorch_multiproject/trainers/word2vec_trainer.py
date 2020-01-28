import os
import pickle
import logging
import torch
from trainers.generic_trainer import GenericTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Word2VecTrainer(GenericTrainer):

    def __init__(self, dataloader_params, subsample_words, *args, **kwargs):
        """
        Trains Skip-Gram Negative Sampling model.
        Parameters
        ----------
        dataloader_params (dict): contains parameters needed for creating an instance of torch Dataloader class.
        subsample_words (bool): if True, the Dataset is resampled and a new Dataloader instance is created every epoch.
        *args: root, model, criterion, optimizer, scheduler, metrics, epochs,
               hyperparams (optional), save_dir (optional), checkpoint (optional),
               change_lr (optional).
        **kwargs: checkpoint (optional).
        """
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.subsample_words = subsample_words

        # create one permanent instance of dataloader if without subsampling
        self.dataloader = None
        if not self.subsample_words:
            dataset = dataloader_params['dataset']
            dataset.subsample_or_get_data()
            self.dataloader = DataLoader(dataset, batch_size=dataloader_params['batch_size'],
                                         shuffle=dataloader_params['shuffle'],
                                         num_workers=dataloader_params['num_workers'])
        else:
            self.dataloader_params = dataloader_params

    def _train_step(self, epoch):
        """
        Behaviour during one pass through the epoch.
        Parameters
        ----------
        epoch (int): current epoch number.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print parameters of optimizer and scheduler every epoch
        self.logger.info(str(self.optimizer))
        if self.scheduler is not None:
            self.logger.info(str(self.scheduler.state_dict()))

        results = {
            'best_performance': False
        }
        cumulative_loss = torch.empty(0)

        if self.subsample_words:
            # resample the data and create a new dataloader on every epoch
            dataset = self.dataloader_params['dataset']
            dataset.subsample_or_get_data()
            self.dataloader = DataLoader(dataset, batch_size=self.dataloader_params['batch_size'],
                                         shuffle=self.dataloader_params['shuffle'],
                                         num_workers=self.dataloader_params['num_workers'])

        self.logger.info('Epoch {}/{}'.format(epoch, self.epochs))
        t = tqdm(iter(self.dataloader), leave=False, total=len(self.dataloader))
        t.set_description("[Epoch {}]".format(epoch))
        for input_word, target_words in t:
            input_word = input_word.to(device)
            target_words = target_words.to(device)

            loss = self.model(input_word, target_words)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t.set_postfix(loss=loss.item())
            running_loss = loss.view(-1).cpu().float()
            cumulative_loss = torch.cat((cumulative_loss, running_loss))

        self.logger.info('Epoch loss: {:.6f}'.format(cumulative_loss.mean()))
        self.logger.info('\n\n')
        # in the current implementation the trained model is saved after each epoch
        results.update({'best_performance': True})
        return results

    def _serialize(self, epoch):
        """
        Saves the model and some other parameters
        Parameters
        ----------
        epoch (int): current epoch number.
        """
        if self.scheduler is not None:
            sched_state = {'name': self.scheduler.__class__.__name__,
                           'state': self.scheduler.state_dict()}
        else:
            sched_state = None

        model_state = {
            'epoch': epoch,
            'model_name': self.name,
            'model_state': self.model.state_dict(),
            'optimizer': {'name': self.optimizer.__class__.__name__,
                          'state': self.optimizer.state_dict()},
            'scheduler': sched_state,
            'best_metrics': self.best_metrics
        }
        chkpt = '{}.pth'.format(self.name)
        model_path = os.path.join(self.save_dir, chkpt)
        idx2vec_path = os.path.join(self.save_dir, 'idx2vec.pickle')
        idx2vec = self.model.input_embeddings.weight.data.cpu().numpy()

        pickle.dump(idx2vec, open(idx2vec_path, 'wb'))
        torch.save(model_state, model_path)

        self.logger.info('Saving the model at {}'.format(model_path))
