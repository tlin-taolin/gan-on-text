# -*- coding: utf-8 -*-
import os
import re
import collections
from os.path import exists, join
from os import listdir
import numpy as np

import code.utils.opfiles as opfile
from code.utils.logger import log
from code.utils.opfiles import build_dirs


class DataLoaderBBC():

    def __init__(self, para):
        """init parameters."""
        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'bbc'

        self.path_data_input = join(data_in_dir, 'text', 'BBC')
        self.path_data_content = join(self.path_data_input, 'all')
        self.path_data_out_dir = join(data_out_dir, self.name)
        self.path_data_out_dict = join(self.path_data_out_dir, 'data.pkl')
        self.path_data_out_string = join(self.path_data_out_dir, 'data.txt')

        # load the data, potentially need to do the preprocessing.
        self.load_data(para.REBUILD_DATA)

        # split the data or restore them from path.
        self.generate_data_for_batch()

        log('the number of sentence is {}, the vocab size is {}'.format(
            self.sentence_size, self.vocab_size))

    """for load data."""
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

        log('...some data statistics.')
        self.sentence_size = self.tensor.shape[0]
        self.sentence_length = self.tensor.shape[1]
        self.vocab_size = len(self.words)
        log('......existing {} sentences, vocabulary size is {}'.format(
            self.sentence_size, self.vocab_size))

    def preprocess(self):
        """preprocess the dataset."""
        log('preprocess the dataset.')

        log('load data.')
        if self.para.REBUILD_DATA or not exists(self.path_data_content):
            log('init content from raw.')

            content = self.init_content()
            self.output_string(
                content, self.path_data_content, delimiter='\n\n')

        log('load context for further preprocessing.')
        content = opfile.read_text_withoutsplit(self.path_data_content)
        sentence_string, sentence_token = self.advance_cleaner(content)

        log('...mask and pad the sentence.')
        padded_sentences_tokens, mask_sentences = self.mask_sentence(
            sentence_token)
        self.vocab, self.words = self.build_vocab(padded_sentences_tokens)
        opfile.write_txt(sentence_string, self.path_data_out_string)

        log('...map word to index.')
        self.tensor = np.array(
            [[self.vocab.get(s) for s in sentence]
             for sentence in padded_sentences_tokens])
        self.mask = np.array(mask_sentences)

        log('...some data statistics.')
        self.sentence_size = self.tensor.shape[0]
        self.sentence_length = self.tensor.shape[1]
        self.vocab_size = len(self.words)
        log('......existing {} sentences, vocabulary size is {}'.format(
            self.sentence_size, self.vocab_size))

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
        string = re.sub(r"\s{2,}", " ", string)

        string = self.basic_cleaner(string)
        sentences = string.split(' . ')
        sentences = [s.split() for s in sentences]
        sentences_tokens = [
            ['<go>'] + filter(lambda x: x != '', s) + ['<eos>']
            for s in sentences]
        string = [' '.join(s) for s in sentences_tokens]
        string = '\n'.join(string)
        return string, sentences_tokens

    def basic_cleaner(self, string):
        """a basic cleaning regex that fixs for all models."""
        string = string.strip().lower()
        for k, v in self.define_search_replacement().items():
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

    def mask_sentence(self, sentences):
        """pad the sentence to a fixed length."""
        sentence_lengths = map(lambda s: len(s), sentences)
        max_len = np.max(sentence_lengths)
        median_len = np.median(sentence_lengths)
        min_len = np.min(sentence_lengths)
        upper = 30
        lower = 10

        log('......max len:{}, median len:{}, min len:{}'.format(
            max_len, median_len, min_len))
        valid_sentences = filter(
            lambda s: len(s) >= lower and len(s) <= upper, sentences)

        padded_sentences = [
            s + ['<pad>'] * (upper - len(s)) for s in valid_sentences]
        mask = [
            [1] * len(s) + [0] * (upper - len(s))
            for s in valid_sentences]
        return padded_sentences, mask

    def build_vocab(self, tokens):
        """build a vocabulary."""
        log('build a vocabulary.')
        log('...flatmap a list of sentence list to a list of sentence.')

        if type(tokens[0]) == list:
            words = []
            for token in tokens:
                words += token
        else:
            words = tokens

        word_counts = collections.Counter(words)

        log('...mapping from index to word.')
        word_list = [x[0] for x in word_counts.most_common()]
        word_list = list(sorted(word_list))

        log('...mapping from word to index.')
        vocab = {x: i for i, x in enumerate(word_list)}
        return vocab, word_list

    """for batch data generation."""
    def determine_batch_pointer_pos(self, stage):
        if 'train' in stage.__name__:
            scope = range(self.num_batches_train)
        elif 'val' in stage.__name__:
            scope = range(self.num_batches_train, self.num_batches)
        else:
            raise NotImplementedError
        return scope

    def generate_data_for_batch(self):
        self.create_batches()
        self.reset_batch_pointer()

    def create_batches(self):
        """create batches for the dataset."""
        if not self.para.DEBUG:
            self.num_batches = int(self.tensor.shape[0] / self.para.BATCH_SIZE)
        else:
            self.num_batches = int(self.para.DEBUG_SIZE / self.para.BATCH_SIZE)

        log('...number of batches: {}'.format(self.num_batches))

        num_samples = self.num_batches * self.para.BATCH_SIZE
        self.tensor = self.tensor[: num_samples]
        self.mask = self.mask[: num_samples, :]
        self.num_batches_train = int(self.num_batches * self.para.RATIO_TRAIN)
        self.num_batches_val = self.num_batches - self.num_batches_train

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

    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        ymask = self.ymask_batches[self.pointer]
        return x, y, ymask

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
