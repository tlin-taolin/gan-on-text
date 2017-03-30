# -*- coding: utf-8 -*-
import re
from os.path import exists, join
import collections

import os
import numpy as np

from code.utils.logger import log
import code.utils.opfiles as opfile
from code.utils.opfiles import build_dirs


class DataLoaderBasic(object):

    def load_data(self):
        """Check if the raw data and processed data exist.
            If not exists, then preprocess and save the data,
            otherwise, load the data directly.
        """
        build_dirs(self.path_data_out_dir)

        if self.para.REBUILD_DATA \
                or not exists(self.path_data_out_dict):
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

    def basic_cleaner(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string

    def generate_data_for_batch(self):
        self.create_batches()
        self.reset_batch_pointer()

    def reset_batch_pointer(self):
        self.pointer = 0

    def output_string(self, data, path_output, delimiter='\n'):
        """join the string in a list and output them to a file."""
        os.remove(path_output) if exists(path_output) else None

        for d in data:
            try:
                opfile.write_txt(d + delimiter, path_output, 'a')
            except:
                print(d)

    def determine_batch_pointer_pos(self, stage, force_total=False):
        self.num_batches_train = int(self.num_batches * self.para.TRAIN_RATIO)
        self.num_batches_val = self.num_batches - self.num_batches_train

        if force_total:
            scope = range(self.num_batches)
        elif 'train' in stage.__name__:
            scope = range(self.num_batches_train)
        elif 'val' in stage.__name__:
            scope = range(self.num_batches_train, self.num_batches)
        else:
            raise NotImplementedError
        return scope

    def swap_random_pos(self, x):
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
