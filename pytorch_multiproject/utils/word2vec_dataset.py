import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import re
import requests
import zipfile
import pickle
import pandas as pd
import numpy as np

data_root = os.path.join(ROOT_DIR, 'resources')


class GetWord2VecData:

    def __init__(self, root, window_size=5, genre='western', url=None, sep_seq=' +++$+++ '):
        self.separator = sep_seq
        self.window_size = window_size
        self.genre = genre
        self.data_dir = root
        self.url = url
        self.unk = '<UNK>'
        if self.url is None:
            self.url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'

    def download_extract_zip(self):
        r = requests.get(self.url)
        local_file_path = os.path.join(self.data_dir, 'word2vec.zip')
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'wb') as f:
            f.write(r.content)

        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(path=self.data_dir)

    def skipgram(self, sentence, idx):
        # generates a pair of input word - target(context) words
        input_word = sentence[idx]
        left = sentence[max(idx - self.window_size, 0): idx]
        right = sentence[idx + 1: idx + 1 + self.window_size]
        # get the list of target or context words for central input word
        target_words = [self.unk for _ in range(self.window_size - len(left))] + left + right + [self.unk for _ in
                                                                                                range(self.window_size - len(right))]
        return input_word, target_words

    def generate_train_data_movies(self, unpacked_name='cornell movie-dialogs corpus'):
        sep_esc = re.escape(self.separator)
        target_folder = os.path.join(self.data_dir, unpacked_name)

        lines = pd.read_csv(os.path.join(target_folder, 'movie_lines.txt'), sep=sep_esc, header=None)
        movie_metadata = pd.read_csv(os.path.join(target_folder, 'movie_titles_metadata.txt'), sep=sep_esc,
                                     header=None)

        # find movie number that correspond to selected genre
        movie_metadata_sel = movie_metadata[movie_metadata[5].str.contains(self.genre)]
        selected_movies = movie_metadata_sel[0].values
        # extract the lines for movies in selected genre
        lines_sel = lines[lines[2].apply(lambda val: val in selected_movies)]
        # 4th column contains dialogue lines
        corpus = lines_sel[4].values

        # convert into char array
        corpus = corpus.astype(str)
        # split sentences into lists of words
        corpus = list(np.char.split(corpus))
        for idx, sentence in enumerate(corpus):
            # convert all to lowercase
            sentence = [word.lower() for word in sentence]
            # strip anything except alphabetic chars
            sentence = [re.sub(r'[^a-z]', '', word) for word in sentence if len(word)> 0]
            # drop all empty strings
            sentence = [word.strip() for word in sentence if word.strip()]
            corpus[idx] = sentence

        # stack lists of sentences into a vector of words
        vocabulary = np.hstack(corpus)
        # get all unique words
        vocabulary = list(set(list(vocabulary)))
        vocabulary.sort()
        # insert an unkown char
        vocabulary.insert(0, self.unk)
        word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        idx2word = {idx: word for idx, word in enumerate(vocabulary)}
        # get a tuple of words and their frequency
        word_counts = np.unique(np.hstack(corpus), return_counts=True)
        # store word counts using word2idx encoding
        word_count_encod = {word2idx[word]: count for word, count in zip(word_counts[0], word_counts[1])}

        # prepare the training pairs of input and context words
        data = []
        for sentence in corpus:
            for idx in range(len(sentence)):
                input_word, target_words = self.skipgram(sentence, idx)
                data.append((word2idx[input_word], [word2idx[word] for word in target_words]))

        # save all the relevant processing results using pickle
        pickle.dump(data, open(os.path.join(target_folder, 'data.pickle'), 'wb'))
        pickle.dump(vocabulary, open(os.path.join(target_folder, 'vocabulary.pickle'), 'wb'))
        pickle.dump(word2idx, open(os.path.join(target_folder, 'word2idx.pickle'), 'wb'))
        pickle.dump(idx2word, open(os.path.join(target_folder, 'idx2word.pickle'), 'wb'))
        pickle.dump(word_count_encod, open(os.path.join(target_folder, 'word_counts.pickle'), 'wb'))


data_getter = GetWord2VecData(data_root)
data_getter.download_extract_zip()
data_getter.generate_train_data_movies()
