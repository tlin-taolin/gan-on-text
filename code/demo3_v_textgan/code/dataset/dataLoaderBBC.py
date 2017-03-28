# -*- coding: utf-8 -*-
import re
import collections
from os.path import exists, join
from os import listdir
import numpy as np

from code.utils.logger import log
import code.utils.opfiles as opfile
from code.dataset.dataLoaderBasic import DataLoaderBasic


class DataLoaderBBC(DataLoaderBasic):

    def __init__(self, para):
        """init parameters."""
        super(DataLoaderBBC, self).__init__()

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

        sentences = string.split(' . ')
        sentences = [s.split() for s in sentences]

        sentences_tokens = [
            ['go'] + filter(lambda x: x != '', s) + ['eos']
            for s in sentences]
        string = [' '.join(s) for s in sentences_tokens]
        string = '\n'.join(string)
        return string, sentences_tokens

    def mask_sentence(self, sentences):
        """pad the sentence to a fixed length."""
        sentence_lengths = map(lambda s: len(s), sentences)
        max_len = np.max(sentence_lengths)
        median_len = np.median(sentence_lengths)
        min_len = np.min(sentence_lengths)
        lower = int(0.6 * self.para.SENTENCE_LENGTH)

        log('......max len:{}, median len:{}, min len:{}'.format(
            max_len, median_len, min_len))
        valid_sentences = filter(
            lambda s: len(s) >= lower and len(s) <= self.para.SENTENCE_LENGTH,
            sentences)

        padded_sentences = [
            s + ['pad'] * (self.para.SENTENCE_LENGTH - len(s))
            for s in valid_sentences]
        mask = [
            [1] * len(s) + [0] * (self.para.SENTENCE_LENGTH - len(s))
            for s in valid_sentences]

        log('......filter sentence and bound them in the range of {}.'.format(
            self.para.SENTENCE_LENGTH))
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
    def create_batches(self):
        """create batches for the dataset."""
        if not self.para.DEBUG:
            self.num_batches = int(self.tensor.shape[0] / self.para.BATCH_SIZE)
        else:
            self.num_batches = int(self.para.DEBUG_SIZE / self.para.BATCH_SIZE)

        log('get data info.')
        self.sentence_size = self.num_batches * self.para.BATCH_SIZE
        self.sentence_length = self.tensor.shape[1]
        self.vocab_size = len(self.words)

        num_samples = self.num_batches * self.para.BATCH_SIZE
        self.tensor = self.tensor[: num_samples]
        self.mask = self.mask[: num_samples, :]
        self.num_batches_train = int(self.num_batches * self.para.TRAIN_RATIO)
        self.num_batches_val = self.num_batches - self.num_batches_train

        log('init batch data.')
        log('...number of batches: {}'.format(self.num_batches))
        x = self.tensor
        y = np.copy(self.tensor)
        ymask = np.copy(self.mask)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = self.vocab['pad'] * np.ones(num_samples)
        z = np.random.uniform(
            size=(self.num_batches * self.para.BATCH_SIZE,
                  self.para.EMBEDDING_SIZE))

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
