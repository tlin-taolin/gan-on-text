# -*- coding: utf-8 -*-
import re
import json
from code.utils.auxiliary import flatten
from os.path import exists, join

import numpy as np


from code.utils.logger import log
from code.utils.auxiliary import make_square
from code.dataset.dataLoaderBasic import DataLoaderBasic


class DataLoaderBBTV1(DataLoaderBasic):
    def __init__(self, para):
        """init parameters."""
        super(DataLoaderBBTV1, self).__init__()

        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'bbt'

        self.path_data_input = join(data_in_dir, 'text', 'BBT')
        self.path_data_out_dir = join(data_out_dir, self.name)

        # load the data, potentially need to do the preprocessing.
        self.load_data()

    def load_data(self):
        """Check if the raw data and processed data exist.
            If not exists, then preprocess and save the data,
            otherwise, load the data directly.
        """
        log('...init bucket.')
        self.buckets, self.bucket_list = \
            self.create_bucket(self.para.BUCKET_OPT)
        self.bucket_noise_list = self.build_noise_bucket()
        self.bucket_dict = self.build_bucket_dict()

        log('...load data.')
        self.init_content()

    def create_bucket(self, options):
        options = map(int, options.split(','))
        self.bucket_options = options
        num_options = len(options)

        buckets = []
        bucket_list = []

        for i in range(num_options):
            buckets.append(options[i])

        for _ in range(len(buckets)):
            bucket_list.append([])
        return buckets, bucket_list

    def build_bucket_dict(self):
        """map different sentence length to the corresponding bucket.
            e.g., we have bucket_options 5,10,15,20,25,31
                sentence length <= 5 will be mapped to the index 0,
                5 < sentence length <= 10 will be mapped to the index 1, etc.
        """
        bucket_dict = {}
        for i in range(1, self.bucket_options[-1] + 1):
            count = len(self.bucket_options) - 1
            for option in reversed(self.bucket_options):
                if option >= i:
                    bucket_dict[i] = count
                count = count - 1
        return bucket_dict

    def init_content(self):
        """load data from text.txt and put them line by line"""
        log('...init data from the raw dataset.')
        self.js_path = join(self.path_data_input, 'dict.json')
        self.ls_path = join(self.path_data_input, 'text.txt')

        log('...read dialogues from file.')
        self.file = open(self.ls_path, 'r')
        self.lines = self.read_txt(self.ls_path, 2)
        self.num_lines = len(self.lines)
        self.line_pointer = 0

        with open(self.js_path, "r") as f:
            self.dict = json.load(f)
            self.vocab_size = len(self.dict)

        self.inv_dict = {}
        for key, value in self.dict.items():
            self.inv_dict[value] = key

    def build_noise_bucket(self):
        if 'uniform' in self.para.NOISE_DISTRIBUTION:
            noises = np.random.uniform(
                size=[self.para.BATCH_SIZE * len(self.bucket_list),
                      self.para.NOISE_DIM])
        elif 'gaussian' in self.para.NOISE_DISTRIBUTION:
            noises = np.random.normal(
                size=[self.para.BATCH_SIZE * len(self.bucket_list),
                      self.para.NOISE_DIM])
        return [
            noises[th * self.para.BATCH_SIZE: (th + 1) * self.para.BATCH_SIZE]
            for th in range(len(self.bucket_list))]

    def next_batch(self):
        # normal mode.
        index = self.fill_bucket()

        if index != -1:
            output = self.bucket_list[index]
            self.bucket_list[index] = []
            noise = self.bucket_noise_list[index]
        else:
            output = None
            noise = None
        return output, noise, index

    def fill_bucket(self):
        while True:
            if self.line_pointer + 1 == self.num_lines:
                self.line_pointer = 0
                return -1
            else:
                self.line_pointer += 1

            line = self.lines[self.line_pointer]
            index = self.check_bucket(line)
            self.bucket_list[index].append(line)

            if len(self.bucket_list[index]) == self.para.BATCH_SIZE:
                return index
        return -1

    def check_bucket(self, line):
        """based on the length of sentences, feed them to a specific bucket."""
        return self.bucket_dict[len(line)]

    def process_bucket(self, data, index):
        bucket = self.buckets[index]
        bucket_len = bucket + 1

        dec_inp = np.zeros((self.para.BATCH_SIZE, bucket_len))
        dec_tar = np.zeros((self.para.BATCH_SIZE, bucket_len))

        # pair[0] = encoder sequence, pair[1] = target sequence
        for i in range(len(data)):
            seq = data[i]

            # copy data to np-array.
            dec_inp[i, 1: len(seq) + 1] = seq
            dec_tar[i, 0: len(seq)] = seq

            # add start/end token
            dec_inp[i, 0] = 2
            dec_tar[i, len(seq)] = 3
        return None, dec_inp, dec_tar

    def read_txt(self, path, size):
        with open(path, 'r') as f:
            data = f.read()
        data = re.findall(r'\[(.*?)\]', data, re.S)
        data = make_square(data, size)
        data = [
            map(lambda x:
                map(int,
                    filter(
                        lambda y: y is not '',
                        x.replace(
                            ']', '').replace('[', '').strip().split(', ')
                        )
                    ),
                instance)
            for instance in data]

        data = flatten(data)

        log('......number of instances: {}'.format(len(data)))
        return data
