# -*- coding: utf-8 -*-
import re
import collections
from os.path import exists

import numpy as np

from code.utils.logger import log
import code.utils.opfiles as opfile
from code.utils.opfiles import build_dirs


class BasicLoader(object):
    """a base model to load the dataset."""

    def __init__(self):
        """init."""
        # system define.
        self.search_replacement = self.define_search_replacement()

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
        self.mask = data['mask']

        self.sentence_size = self.tensor.shape[0]
        self.sentence_length = self.tensor.shape[1]
        self.vocab_size = len(self.words)

    def build_vocab(self, tokens):
        """build a vocabulary."""
        log('build a vocabulary.')
        log('...flatmap a list of sentence list to a list of sentence.')

        words = []
        for token in tokens:
            words += token

        word_counts = collections.Counter(words)

        log('...mapping from index to word.')
        word_list = [x[0] for x in word_counts.most_common()]
        word_list = list(sorted(word_list))

        log('...mapping from word to index.')
        vocab = {x: i for i, x in enumerate(word_list)}
        return vocab, word_list

    def basic_cleaner(self, string):
        """a basic cleaning regex that fixs for all models."""
        string = string.strip().lower()
        for k, v in self.search_replacement.items():
            string = string.replace(k, v)

        # remove dot after single capital letter
        string = re.sub('\$(\d+)', '\1 dollars', string)
        string = re.sub('£(\d+)', '\1 pounds', string)
        string = re.sub(
            '\$(\d+)\.([0-9])\smillion',
            '\1 point \2 million dollars', string)
        string = re.sub('\(i\)', '(1),', string)
        string = re.sub('\(ii\)', '(2),', string)
        string = re.sub('\(iii\)', '(3),', string)
        string = re.sub('\(iv\)', '(4),', string)
        string = re.sub('\(v\)', '(5),', string)
        string = re.sub('\(vi\)', '(6),', string)
        string = re.sub('\(vii\)', '(7),', string)
        string = re.sub('\(viii\)', '(8),', string)
        string = re.sub('\(ix\)', '(9),', string)

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

    def generate_data_for_batch(self):
        self.create_batches()
        self.reset_batch_pointer()

    def create_batches(self):
        """create batches for the dataset."""
        if not self.para.DEBUG:
            self.num_batches = int(self.tensor.shape[0] / self.para.BATCH_SIZE)
        else:
            self.num_batches = int(self.para.DEBUG_SIZE / self.para.BATCH_SIZE)

        num_samples = self.num_batches * self.para.BATCH_SIZE

        self.tensor = self.tensor[: num_samples, :]
        self.mask = self.mask[: num_samples, :]
        self.z = self.para.Z_PRIOR(size=(num_samples, self.para.Z_DIM))

        x = self.tensor
        y = np.copy(self.tensor)
        ymask = np.copy(self.mask)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = self.vocab['<pad>'] * np.ones(num_samples)

        if self.para.SHUFFLE_DATA:
            shuffle_indices = np.random.permutation(num_samples)
            shuffled_x = x[shuffle_indices]
            shuffled_y = y[shuffle_indices]
            shuffled_ymask = ymask[shuffle_indices]
        else:
            shuffled_x = x
            shuffled_y = y
            shuffled_ymask = ymask

        self.x_batches = [
            shuffled_x[
                batch_th * self.para.BATCH_SIZE:
                (batch_th + 1) * self.para.BATCH_SIZE]
            for batch_th in range(self.num_batches)]

        self.z_batches = [
            self.z[
                batch_th * self.para.BATCH_SIZE:
                (batch_th + 1) * self.para.BATCH_SIZE]
            for batch_th in range(self.num_batches)]

        self.y_batches = [
            shuffled_y[
                batch_th * self.para.BATCH_SIZE:
                (batch_th + 1) * self.para.BATCH_SIZE]
            for batch_th in range(self.num_batches)]

        self.ymask_batches = [
            shuffled_ymask[
                batch_th * self.para.BATCH_SIZE:
                (batch_th + 1) * self.para.BATCH_SIZE]
            for batch_th in range(self.num_batches)]

    def swap_x(self, x):
        row, col = x.shape
        random_choice_1 = np.random.randint(col, size=row)
        random_choice_2 = np.remainder(
            random_choice_1 + np.random.randint(1, high=col, size=row), col)

        x = np.copy(x)
        tmp = np.copy(x[np.arange(row), random_choice_1])
        x[np.arange(row), random_choice_1] = np.copy(
            x[np.arange(row), random_choice_2])
        x[np.arange(row), random_choice_1] = tmp
        return x

    def next_batch(self):
        x = self.x_batches[self.pointer]
        z = self.z_batches[self.pointer]
        y = self.y_batches[self.pointer]
        ymask = self.ymask_batches[self.pointer]
        self.pointer = np.mod(self.pointer + 1, self.num_batches)
        return x, z, y, ymask

    def reset_batch_pointer(self):
        self.pointer = 0
