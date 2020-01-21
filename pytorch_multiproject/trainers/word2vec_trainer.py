import os
import pickle
import logging
import torch
from trainers.generic_trainer import GenericTrainer
from tqdm import tqdm


class Word2VecTrainer(GenericTrainer):

    def __init__(self, dataloader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.dataloader = dataloader

    def _train_step(self, epoch):

        # print parameters of optimizer and scheduler every epoch
        self.logger.info(str(self.optimizer))
        if self.scheduler is not None:
            self.logger.info(str(self.scheduler.state_dict()))

        results = {
            'best_performance': False
        }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        t = tqdm(iter(self.dataloader), leave=False, total=len(self.dataloader))
        t.set_description("[Epoch {}]".format(epoch))
        for input_word, target_words in t:
            input_word = input_word.to(device)
            target_words = target_words.to(device)

            loss = self.model(input_word, target_words)
            loss.backward()
            self.optimizer.step()
            t.set_postfix(loss=loss.item())

        # in the current implementation the trained model is saved after each epoch
        results.update({'best_performance': True})
        return results

    def _serialize(self, epoch):
        # save the model and some other parameters
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

        self.logger.info('Saving the model at {}'.format(model_state))
