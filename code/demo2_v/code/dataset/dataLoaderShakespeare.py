# -*- coding: utf-8 -*-
import os
import re
from os.path import exists, join
from os import listdir
import functools
import collections

import numpy as np

import code.utils.opfiles as opfile
from code.utils.logger import log
from code.utils.opfiles import build_dirs


class DataLoaderShakespeare():

    def __init__(self, para):
        """init parameters."""
        super(DataLoaderShakespeare, self).__init__()

        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'shakespeare'

        data_in_dir = join(data_in_dir, 'text', 'TinyShakespeare')
        self.path_data_input = join(data_in_dir, 'input.txt')
        self.path_data_out_dir = join(data_out_dir, self.name)
        self.path_data_out_dict = join(self.path_data_out_dir, 'data.pkl')
        self.path_data_out_string = join(self.path_data_out_dir, 'data.txt')

        # load the data, potentially need to do the preprocessing.
        self.load_data(para.REBUILD_DATA)

        # split the data or restore them from path.
        self.generate_data_for_batch()

        log('the number of sentence is {}, the vocab size is {}'.format(
            self.sentence_size, self.vocab_size))

    def load_data(self, force=False):
        """Check if the raw data and processed data exist.
            If not exists, then preprocess and save the data,
            otherwise, load the data directly.
        """
        build_dirs(self.path_data_out_dir)

        if force or not exists(self.path_data_out_dict):
            log("reading and processing the text file.")
            self.preprocess()
        else:
            log("loading preprocessed files.")
            self.load_preprocessed()

    def load_preprocessed(self):
        data = opfile.load_cpickle(self.path_data_out_dict)
        self.vocab = data['vocab']
        self.words = data['words']
        self.tensor = data['tensor']

        self.vocab_size = len(self.words)
        self.sentence_length = 10

        log('......existing {} words, vocabulary size is {}'.format(
            len(self.tensor), self.vocab_size))

    def preprocess(self):
        """preprocess the dataset."""
        log('preprocess the dataset.')

        with open(self.path_data_input, "r") as f:
            data = f.read()
            sentence_string, sentence_token = self.advance_cleaner(data)

        opfile.output_string(sentence_string, self.path_data_out_string)

        self.vocab, self.words = self.build_vocab(sentence_token)

        log('...map word to index.')
        self.tensor = np.array([self.vocab.get(s) for s in sentence_token])

        log('...some data statistics.')
        self.vocab_size = len(self.words)
        self.seq_length = 10
        log('......existing {} words, vocabulary size is {}'.format(
            len(self.tensor), self.vocab_size))

        log('...save processed data to file.')
        data = {
            'vocab': self.vocab,
            'words': self.words,
            'tensor': self.tensor
        }
        opfile.write_cpickle(data, self.path_data_out_dict)

    def advance_cleaner(self, string):
        """advance cleaner, normally specifically for this dataset."""
        string = self.basic_cleaner(string)
        string = string.strip().lower()
        return string, string.split()

    def basic_cleaner(self, string):
        """a basic cleaning regex that fixs for all models."""
        string = string.strip().lower()
        for k, v in self.search_replacement.items():
            string = string.replace(k, v)

        # remove dot after single capital letter
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

        string = re.sub(r"!", " ! ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r';', ' ; ', string)
        string = re.sub(r'\.', ' . ', string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r'"', '', string)
        return string

    def define_search_replacement(self):
        """define a dictionary for search replacement."""
        return {
              'è': 'e', 'ł': 'l', 'á': 'a', 'Á': 'A', 'â': 'a', 'â': 'a',
              'Â': 'A', 'à': 'a', 'À': 'A', 'å': 'a', 'Å': 'A', 'ã': 'a',
              'Ã': 'A', 'ä': 'a', 'Ä': 'A', 'æ': 'ae', 'Æ': 'ae', 'ç': 's',
              'Ç': 'S', 'ð': 'o', 'Ð': 'D', 'é': 'e', 'É': 'E', 'ê': 'e',
              'Ê': 'E', 'è': 'e', 'È': 'E', 'ë': 'e', 'Ë': 'E', 'í': 'i',
              'Í': 'I', 'î': 'i', 'Î': 'I', 'ì': 'i', 'Ì': 'I', 'ï': 'i',
              'Ï': 'I', 'ñ': 'ni', 'Ñ': 'N', 'ó': 'o', 'Ó': 'O', 'ô': 'o',
              'Ô': 'O', 'ò': 'o', 'Ò': 'o', 'ø': 'o', 'Ø': 'O', 'õ': 'o',
              'Õ': 'O', 'ö': 'o', 'Ö': 'O', 'ß': 'ss', 'þ': 'b', 'š': 's',
              'Š': 'S', 'Þ': 'b', 'ú': 'u', 'Ú': 'U', 'û': 'u', 'Û': 'U',
              'ù': 'u', 'Ù': 'U', 'ü': 'u', 'Ü': 'U', 'Ō': 'O', 'ō': 'o',
              'ý': 'y', 'Ý': 'Y', 'ÿ': 'y'
        }

    def build_vocab(self, tokens):
        """build a vocabulary."""
        log('build a vocabulary.')
        log('...flatmap a list of sentence list to a list of sentence.')

        words = tokens
        word_counts = collections.Counter(words)

        log('...mapping from index to word.')
        word_list = [x[0] for x in word_counts.most_common()]
        word_list = list(sorted(word_list))

        log('...mapping from word to index.')
        vocab = {x: i for i, x in enumerate(word_list)}
        return vocab, word_list

    def generate_data_for_batch(self):
        self.create_batches()
        self.reset_batch_pointer()

    def create_batches(self):
        """create batches for the dataset."""
        self.sentence_length = self.para.SENTENCE_LENGTH

        if not self.para.DEBUG:
            self.num_batches = int(
                self.tensor.shape[0] / self.para.BATCH_SIZE /
                self.sentence_length)
        else:
            self.num_batches = int(
                self.para.DEBUG_SIZE / self.para.BATCH_SIZE /
                self.sentence_length)

        log('...number of batches: {}'.format(self.num_batches))

        self.sentence_size = self.num_batches * self.para.BATCH_SIZE * self.sentence_length
        self.tensor = self.tensor[: self.sentence_size]
        self.num_batches_train = int(self.num_batches * self.para.RATIO_TRAIN)
        self.num_batches_val = self.num_batches - self.num_batches_train

        x = self.tensor
        y = np.copy(self.tensor)
        y[:-1] = x[1:]
        y[-1] = x[0]
        self.x_batches = np.split(
            x.reshape(self.para.BATCH_SIZE, -1), self.num_batches, 1)
        self.y_batches = np.split(
            y.reshape(self.para.BATCH_SIZE, -1), self.num_batches, 1)

    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
