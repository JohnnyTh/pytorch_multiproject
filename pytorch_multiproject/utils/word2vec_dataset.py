import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import re
import shutil
import pickle
import codecs
import requests
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prep_lotr', type=bool, help='prepares the LOTR dataset')
    parser.add_argument('-file_path', type=str, help="full path to a downloaded dataset file")
    parser.add_argument('--url', type=str, default=None, help="url with target dataset file to download")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--min_sentence_len', type=int, default=5,
                        help="minimum length of sentences in processed corpus")
    parser.add_argument('--min_freq', type=int, default=5, help="minimum frequency of the words in vocabulary")
    parser.add_argument('--accumulate_corpus', type=bool, default=True,
                        help="whether the processed corpus will be stored in RAM or loaded from a disc")
    return parser.parse_args()


data_root = os.path.join(ROOT_DIR, 'resources')


class GetWord2VecData:

    def __init__(self, root, window_size=5, url=None, min_sentence_len=5, min_freq=5,
                 accumulate_corpus=False):
        """
        Downloads and pre-processes a corpus of text for Word2Vec algorithm.
        Parameters
        ----------
        root (str): root dir address where all the files will be downloaded and saved after pre-processing.
        window_size (int): context window size for Skip-Gram model.
        url (str): url to download the dataset from.
        min_sentence_len (int): sentences with length below this value will be discarded from corpus.
        min_freq (int): words encountered less then this number of time will be discarded from vocabulary.
        accumulate_corpus (bool): if False, the preprocessed read from the disc and preprocessed two times:
            first time to obtain vocabluary, word counts and idx - word mapping dictionaries and second time to form
            input - context word pairs. Else the corpus will be kept in memory after the initial preprocessing.
        """
        self.data_dir = root
        self.window_size = window_size
        self.url = url
        self.min_sentence_len = min_sentence_len
        self.min_freq = min_freq
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        self.accumulate_corpus = accumulate_corpus
        self.corpus = []
        self.vocabulary = None
        self.word2idx = None
        self.idx2word = None
        self.word_count = None

    def download_txt(self):
        # download a txt file from url.
        print('Accessing the dataset at {}'.format(self.url))
        r = requests.get(self.url)
        local_file_path = os.path.join(self.data_dir, 'word2vec.txt')
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'wb') as f:
            f.write(r.content)
        print('Dataset downloaded'.format(self.url))
        print('Done')

    def skipgram(self, sentence, idx):
        """
        Creates input - target word pair from sentence.
        Parameters
        ----------
        sentence (list): a sequence of words forming a sentence.
        idx (int): index of input word around which the context mush be sampled.

        Returns
        -------
        input_word, target_words - input - target word pair.
        """
        # generates a pair of input word - target(context) words
        input_word = sentence[idx]
        left = sentence[max(idx - self.window_size, 0): idx]
        right = sentence[idx + 1: idx + 1 + self.window_size]
        # get the list of target or context words for central input word
        target_words = [self.unk for _ in range(self.window_size - len(left))] + left + right + \
                       [self.unk for _ in range(self.window_size - len(right))]
        return input_word, target_words

    def build_tools(self, file_path):
        """
        Pre-processes the corpus, creates and saves vocabluary, word counts and idx - word mapping dictionaries.
        Parameters
        ----------
        file_path (str): a path to the corpus of text.
        """
        print("building vocab...")
        num_lines = sum(1 for line in open(file_path, 'r', errors='ignore'))
        word_count = {}
        with codecs.open(file_path, 'r', 'ascii', errors='ignore') as file:
            t = tqdm(file, total=num_lines)
            t.set_description('Processing lines')
            for line in t:
                line = line.lower()
                if not line:
                    continue
                sentence = line.split()
                sentence = [re.sub(r'[^a-z]', '', word) for word in sentence if len(word) > 0]
                sentence = [word.strip() for word in sentence if word.strip()]
                if self.accumulate_corpus and len(sentence) > self.min_sentence_len:
                    self.corpus.append(sentence)

                for word in sentence:
                    word_count[word] = word_count.get(word, 0) + 1
        print("")
        word_count = {key: value for key, value in word_count.items() if value > self.min_freq}

        self.vocabulary = sorted(set(word_count.keys()))
        self.vocabulary.insert(0, self.unk)
        self.vocabulary.insert(0, self.pad)

        self.idx2word = {idx: word for idx, word in enumerate(self.vocabulary)}
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.word_count = {self.word2idx[word]: count for word, count in word_count.items()}
        print('Vocab size: {} unique words'.format(len(self.vocabulary)))
        save_dir = os.path.dirname(file_path)
        file_names = ['vocabulary', 'word2idx', 'idx2word', 'word_counts']
        files = [self.vocabulary, self.word2idx, self.idx2word, self.word_count]
        self.pickle_dump(file_names, files, save_dir)

    def convert_data_from_file(self, file_path):
        """
        Generates input -context pairs from the file saved on disc.
        Parameters
        ----------
        file_path (str): a path to the corpus of text.
        """
        print("converting corpus...")
        data = []
        num_lines = sum(1 for line in open(file_path, 'r', errors='ignore'))
        with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            t = tqdm(file, total=num_lines)
            t.set_description('Preparing input-context pairs')
            for line in file:
                line = line.lower()
                if not line:
                    continue
                sentence = []
                line = line.split()
                line = [re.sub(r'[^a-z]', '', word) for word in line if len(word) > 0]
                line = [word.strip() for word in line if word.strip()]
                for word in line:
                    if word in self.vocabulary:
                        sentence.append(word)
                    else:
                        sentence.append('<UNK>')
                for i in range(len(sentence)):
                    input_word, target_words = data_getter.skipgram(sentence, i)
                    data.append((self.word2idx[input_word], [self.word2idx[t_word] for t_word in target_words]))
                    t.set_postfix('{} input-target pairs created so far'.format(len(data)))
        print("")
        print('{} of input-target(content) pairs have been formed'.format(len(data)))
        self.pickle_dump(['data'], [data], os.path.dirname(file_path))

    def convert_data_from_buffer(self, file_path):
        """
        Generates input -context pairs from the corpus saved in memory.
        Parameters
        ----------
        file_path (str): a path to a directory where the results will be saved.
        """
        print("converting corpus...")
        data = []
        t = tqdm(self.corpus)
        t.set_description('Preparing input-context pairs')
        for line in t:
            sentence = []
            for word in line:
                if word in self.vocabulary:
                    sentence.append(word)
                else:
                    sentence.append('<UNK>')
            for i in range(len(sentence)):
                input_word, target_words = data_getter.skipgram(sentence, i)
                data.append((self.word2idx[input_word], [self.word2idx[t_word] for t_word in target_words]))

                t.set_postfix(progress='{} input -target pairs created so far'.format(len(data)))
        print("")
        print('{} of input - target(content) pairs have been formed'.format(len(data)))
        self.pickle_dump(['data'], [data], os.path.dirname(file_path))

    def generate_data_large(self, file_path):
        # since method generate_train_data_movies() works with all the data loaded in memory at once,
        # for big datasets we need to use a different approach and collect and covert the necessary data as we go
        self.build_tools(file_path)
        if self.accumulate_corpus is False:
            self.convert_data_from_file(file_path)
        else:
            self.convert_data_from_buffer(file_path)

    @staticmethod
    def split_lines_subsentences(corpus):
        # split lines that contain multiple sentences into sub-sentences
        corpus_mod = []
        for sentence in corpus:
            regex_patt = '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
            sentence_splt = re.split(regex_patt, sentence)
            if len(sentence_splt) > 1:
                for el in sentence_splt:
                    corpus_mod.append(el)
            else:
                corpus_mod.append(sentence_splt[0])

        return corpus_mod

    @staticmethod
    def pickle_dump(file_names, files, target_folder):
        """
        Saves the provided objects under respective file names.
        Parameters
        ----------
        file_names (list): a sequence of file name strings.
        files (list): a sequnce of objects to save in pickled form.
        target_folder (str): a path to the save dir.
        """
        for name, content in zip(file_names, files):
            os.makedirs(os.path.join(target_folder, name), exist_ok=True)
            path = os.path.join(os.path.join(target_folder, name),  '{}.pickle'.format(name))
            pickle.dump(content, open(path, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    data_getter = GetWord2VecData(data_root, url=args.url, window_size=args.window,
                                  min_sentence_len=args.min_sentence_len, min_freq=args.min_freq,
                                  accumulate_corpus=args.accumulate_corpus)
    if args.prep_lotr:
        path_src = '/content/drive/My Drive/Colab Notebooks/pt_proj/embeddings/files_data/LOTR_3.txt'
        path_tgt = os.path.join(data_root, 'LOTR_3.txt')
        os.makedirs(os.path.dirname(path_tgt), exist_ok=True)
        shutil.copy2(path_src, path_tgt)
        data_getter.generate_data_large(path_tgt)
    else:
        data_getter.download_txt()
        data_getter.generate_data_large(args.file_path)
