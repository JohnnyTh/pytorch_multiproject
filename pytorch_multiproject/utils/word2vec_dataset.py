import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import re
import codecs
import requests
import zipfile
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--window', type=int, default=5, help="window size")
    return parser.parse_args()


data_root = os.path.join(ROOT_DIR, 'resources')


class GetWord2VecData:

    def __init__(self, root, sep_seq=' +++$+++ ', window_size=5, genre='sci-fi', url=None,
                 min_sentence_len=5, min_freq=5):
        self.data_dir = root
        self.separator = sep_seq
        self.window_size = window_size
        self.genre = genre
        self.url = url
        self.min_sentence_len = min_sentence_len
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        if self.url is None:
            self.url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
        self.min_freq = min_freq
        self.vocabulary = None
        self.word2idx = None
        self.idx2word = None
        self.word_count = None

    def download_extract_zip(self):
        print('Accessing the dataset at {}'.format(self.url))
        r = requests.get(self.url)
        local_file_path = os.path.join(self.data_dir, 'word2vec.zip')
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'wb') as f:
            f.write(r.content)
        print('Dataset downloaded'.format(self.url))

        print('Unpacking the data in {}'.format(self.data_dir))
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(path=self.data_dir)
        print('Done')

    def skipgram(self, sentence, idx):
        # generates a pair of input word - target(context) words
        input_word = sentence[idx]
        left = sentence[max(idx - self.window_size, 0): idx]
        right = sentence[idx + 1: idx + 1 + self.window_size]
        # get the list of target or context words for central input word
        target_words = [self.unk for _ in range(self.window_size - len(left))] + left + right + [self.unk for _ in
                                                                                                range(self.window_size - len(right))]
        return input_word, target_words

    def build_tools(self, file_path):
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

    def convert_data(self, file_path):
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
        print("")
        print('{} of input - target(content) pairs have been formed'.format(len(data)))
        self.pickle_dump(['data'], [data], os.path.basename(file_path))

    def generate_data_large(self, file_path):
        # since method generate_train_data_movies() works with all the data loaded in memory at once,
        # for big datasets we need to use a different approach and collect and covert the necessary data as we go
        save_dir = os.path.basename(file_path)
        self.build_tools(file_path)
        file_names = ['vocabulary', 'word2idx', 'idx2word', 'word_counts']
        files = [self.vocabulary, self.word2idx, self.idx2word, self.word_count]
        self.pickle_dump(file_names, files, save_dir)

        self.convert_data()

    def generate_train_data_movies(self, file_folder='cornell movie-dialogs corpus'):
        sep_esc = re.escape(self.separator)
        target_folder = os.path.join(self.data_dir, file_folder)

        lines = pd.read_csv(os.path.join(target_folder, 'movie_lines.txt'), sep=sep_esc, header=None)
        movie_metadata = pd.read_csv(os.path.join(target_folder, 'movie_titles_metadata.txt'), sep=sep_esc,
                                     header=None)

        # find movie number that correspond to selected genre
        movie_metadata_sel = movie_metadata[movie_metadata[5].str.contains(self.genre)]
        selected_movies = movie_metadata_sel[0].values
        # extract the lines for movies in selected genre
        lines_sel = lines[lines[2].apply(lambda val: val in selected_movies)]
        # 4th column contains dialogue lines
        corpus = list(lines_sel[4].values.astype(str))

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

        # split sentences into words
        corpus_mod = list(np.char.split(corpus_mod))
        for idx, sentence in enumerate(corpus_mod):
            # convert all to lowercase
            sentence = [word.lower() for word in sentence]
            # strip anything except alphabetic chars
            sentence = [re.sub(r'[^a-z]', '', word) for word in sentence if len(word) > 0]
            # drop all empty strings
            sentence = [word.strip() for word in sentence if word.strip()]
            corpus_mod[idx] = sentence
        # discard sentences that have length less than specified
        corpus_mod = [sentence for sentence in corpus_mod if len(sentence) >= self.min_sentence_len]

        word_counts = np.unique(np.hstack(corpus_mod), return_counts=True)
        # drop words that occur less than specified
        counts_mask = word_counts[1] > self.min_freq
        word_counts = (word_counts[0][counts_mask], word_counts[1][counts_mask])
        vocabulary = list(word_counts[0])
        # insert an unknown char at 1, 0 is reserved for padding word
        vocabulary.insert(0, self.unk)
        vocabulary.insert(0, self.pad)
        # create word - idx conversion dictionaries
        word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        idx2word = {idx: word for idx, word in enumerate(vocabulary)}
        # store word counts using word2idx encoding
        word_count_encod = {word2idx[word]: count for word, count in zip(word_counts[0], word_counts[1])}

        # prepare the training pairs of input and context words
        data = []
        t = tqdm(corpus_mod)
        print('Creating input-context word pairs')
        t.set_description('Processing sentence')
        for sentence in t:
            # replace unknown words in sentence with respective token
            sent_unk = []
            for word in sentence:
                if word in vocabulary:
                    sent_unk.append(word)
                else:
                    sent_unk.append(self.unk)
            for idx in range(len(sentence)):
                input_word, target_words = self.skipgram(sent_unk, idx)
                data.append((word2idx[input_word], [word2idx[word] for word in target_words]))

        print('Dataset preprocessing finished')
        print('Vocab size: {} unique words'.format(len(vocabulary)))
        print('{} of input - target(content) pairs have been formed'.format(len(data)))
        print('Saving the pickled data...')

        # save all the relevant processing results using pickle
        var_names = ['data', 'vocabulary', 'word2idx', 'idx2word', 'word_counts']
        to_dump = [data, vocabulary, word2idx, idx2word, word_count_encod]
        self.pickle_dump(var_names, to_dump, target_folder)
        print('Done')

    @staticmethod
    def pickle_dump(file_names, files, target_folder):
        for name, content in zip(file_names, files):
            os.makedirs(os.path.join(target_folder, name), exist_ok=True)
            path = os.path.join(os.path.join(target_folder, name),  '{}.pickle'.format(name))
            pickle.dump(content, open(path, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    data_getter = GetWord2VecData(data_root)
    # preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    # preprocess.build(args.vocab, max_vocab=args.max_vocab)
    # preprocess.convert(args.corpus)

# data_getter.download_extract_zip()
# data_getter.generate_train_data_movies()
