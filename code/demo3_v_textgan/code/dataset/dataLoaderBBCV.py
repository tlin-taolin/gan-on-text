# -*- coding: utf-8 -*-
import re
import collections
from os.path import exists, join
from os import listdir
import numpy as np

from code.utils.logger import log
import code.utils.opfiles as opfile
from code.dataset.dataLoaderBasic import DataLoaderBasic


class DataLoaderBBCV(DataLoaderBasic):

    def __init__(self, para):
        """init parameters."""
        super(DataLoaderBBCV, self).__init__()

        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'bbcv'

        self.path_data_input = join(data_in_dir, 'text', 'BBC')
        self.path_data_content = join(self.path_data_input, 'all')
        self.path_data_out_dir = join(data_out_dir, self.name)
        self.path_data_out_dict = join(self.path_data_out_dir, 'data.pkl')

        # load the data, potentially need to do the preprocessing.
        self.load_data()

        # split the data or restore them from path.
        self.generate_data_for_batch()

        log('num of sentence: {}, sentence length: {}, vocab size: {}'.format(
            self.sentence_size, self.sentence_length, self.vocab_size))

    """preprocess data."""
    def preprocess(self):
        """preprocess the dataset."""
        log('preprocess the dataset.')

        log('load data.')
        if not exists(self.path_data_content):
            log('init content from raw.')

            content = self.init_content()
            self.output_string(
                content, self.path_data_content, delimiter='\n\n')

        log('load context for further preprocessing.')
        content = opfile.read_text_withoutsplit(self.path_data_content)
        sentence_string, sentence_token = self.advance_cleaner(content)

        log('...build vocab and map word to index.')
        self.vocab, self.words = self.build_vocab(sentence_token)
        self.tensor = np.array([self.vocab.get(s) for s in sentence_token])
        self.mask = np.ones_like(self.tensor)

        log('...save processed data to file.')
        data = {
            'vocab': self.vocab,
            'words': self.words,
            'tensor': self.tensor,
            'mask': self.mask
        }
        opfile.write_cpickle(data, self.path_data_out_dict)

    def init_content(self):
        """load data from path and remove some useless information."""
        log('init data from the raw dataset.')

        content = []
        sub_folders = [
            'business', 'entertainment', 'politics', 'sport', 'tech']

        for sub_folder in sub_folders:
            sub_path = join(self.path_data_input, sub_folder)

            for sub_file in listdir(sub_path):
                tmp_path = join(sub_path, sub_file)
                tmp_data = opfile.read_txt(tmp_path)
                tmp_data = [x for x in tmp_data[1:] if x != '']
                content.append(' '.join(tmp_data))
        return content

    def advance_cleaner(self, string):
        """advance cleaner, normally specifically for this dataset."""
        log('clean data.')

        string = string.strip().lower()
        string = re.sub("\(.*?\)", "", string)
        string = re.sub("\d,\d+", "", string)
        string = re.sub("\n", " ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"\s{2,}", " ", string)

        if self.para.CLEAN_DATA:
            string = self.basic_cleaner(string)
        return string, string.split()

    def build_vocab(self, tokens):
        """build a vocabulary."""
        log('build a vocabulary.')
        log('...flatmap a list of sentence list to a list of sentence.')

        words = tokens
        word_counts = collections.Counter(words)

        log('...mapping from index to word.')
        word_list = [x[0] for x in word_counts.most_common()]
        word_list = list(sorted(word_list))

        log('...add additional <go> and <eos>.')
        word_list.append('go')
        word_list.append('eos')

        log('...mapping from word to index.')
        vocab = {x: i for i, x in enumerate(word_list)}
        return vocab, word_list

    """for batch data generation."""
    def create_batches(self):
        """create batches for the dataset."""
        if not self.para.DEBUG:
            self.num_batches = int(
                self.tensor.shape[0] / self.para.BATCH_SIZE /
                self.para.SENTENCE_LENGTH)
        else:
            self.num_batches = int(
                self.para.DEBUG_SIZE / self.para.BATCH_SIZE /
                self.para.SENTENCE_LENGTH)

        log('get data info.')
        self.vocab_size = len(self.words)
        self.sentence_length = self.para.SENTENCE_LENGTH
        self.sentence_size = self.num_batches * self.para.BATCH_SIZE

        words_size = self.num_batches * self.para.BATCH_SIZE * self.para.SENTENCE_LENGTH
        self.tensor = self.tensor[: words_size]
        self.mask = self.mask[: words_size]

        log('init batch data.')
        log('...number of batches: {}'.format(self.num_batches))
        x = self.tensor
        y = np.copy(self.tensor)
        y[:-1] = x[1:]
        y[-1] = x[0]
        ymask = np.copy(self.mask)
        z = np.random.uniform(
            size=(self.num_batches * self.para.BATCH_SIZE,
                  self.para.EMBEDDING_SIZE))

        self.x_batches = np.split(
            x.reshape(self.para.BATCH_SIZE, -1), self.num_batches, 1)
        self.y_batches = np.split(
            y.reshape(self.para.BATCH_SIZE, -1), self.num_batches, 1)
        self.ymask_batches = np.split(
            ymask.reshape(self.para.BATCH_SIZE, -1), self.num_batches, 1)
        self.z_batches = [
            z[batch_th * self.para.BATCH_SIZE:
              (batch_th + 1) * self.para.BATCH_SIZE]
            for batch_th in range(self.num_batches)]

    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        ymask = self.ymask_batches[self.pointer]
        z = self.z_batches[self.pointer]
        self.pointer += 1
        return x, y, ymask, z
