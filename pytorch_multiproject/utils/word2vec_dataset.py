import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
sys.path.append(ROOT_DIR)
import re
import logging
import requests
import zipfile
import pickle
import pandas as pd
import numpy as np

data_root = os.path.join(ROOT_DIR, 'resources')


class GetWord2VecData:

    def __init__(self, root, sep_seq=' +++$+++ ', window_size=5, genre='sci-fi', url=None, min_sentence_len=5, ):
        self.data_dir = root
        self.separator = sep_seq
        self.window_size = window_size
        self.genre = genre
        self.url = url
        self.min_sentence_len = min_sentence_len
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        self.logger = logging.getLogger(os.path.basename(__file__))
        if self.url is None:
            self.url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'

    def download_extract_zip(self):
        self.logger.info('Accessing the dataset at {}'.format(self.url))
        r = requests.get(self.url)
        local_file_path = os.path.join(self.data_dir, 'word2vec.zip')
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, 'wb') as f:
            f.write(r.content)
        self.logger.info('Dataset downloaded'.format(self.url))

        self.logger.info('Unpacking the data in {}'.format(self.data_dir))
        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(path=self.data_dir)
        self.logger.info('Done')

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
        # drop words that occur less than 5 times
        counts_mask = word_counts[1] > 5
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
        for sentence in corpus:
            # replace unknown words in sentence with respective token
            sent_unk = []
            for word in sentence:
                if word in vocabulary:
                    sent_unk.append(word)
                else:
                    sent_unk.append('<UNK>')
            for idx in range(len(sentence)):
                input_word, target_words = self.skipgram(sent_unk, idx)
                data.append((word2idx[input_word], [word2idx[word] for word in target_words]))

        self.logger.info('Dataset preprocessing finished')
        self.logger.info('Vocab size: {} unique words'.format(len(vocabulary)))
        self.logger.info('{} of input - target(content) pairs have been formed'.format(len(data)))
        self.logger.info('Saving the pickled data...')

        # save all the relevant processing results using pickle
        var_names = ['data', 'vocabulary', 'word2idx', 'idx2word', 'word_counts']
        to_dump = [data, vocabulary, word2idx, idx2word, word_count_encod]
        for name, content in zip(var_names, to_dump):
            path = os.path.join(os.path.join(target_folder, name),  '{}.pickle'.format(name))
            pickle.dump(content, open(path, 'wb'))
        self.logger.info('Done')


data_getter = GetWord2VecData(data_root)
data_getter.download_extract_zip()
data_getter.generate_train_data_movies()
