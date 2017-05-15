# -*- coding: utf-8 -*-
import re
import json
import collections
from os.path import exists, join
import numpy as np

from code.utils.logger import log
import code.utils.opfiles as opfile
from code.dataset.dataLoaderBasic import DataLoaderBasic


class DataLoaderBBT(DataLoaderBasic):
    def __init__(self, para):
        """init parameters."""
        super(DataLoaderBBT, self).__init__()

        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'bbt'

        self.path_data_input = join(data_in_dir, 'text', 'BBT')
        self.path_data_content = join(self.path_data_input, 'all')
        self.path_data_out_sent = join(self.path_data_input, 'sent.pkl')
        self.path_data_out_dir = join(data_out_dir, self.name)
        self.path_data_out_dict = join(self.path_data_out_dir, 'data.pkl')
        self.path_data_out_string = join(self.path_data_out_dir, 'data.txt')

        # load the data, potentially need to do the preprocessing.
        self.load_data()

        # split the data or restore them from path.
        self.generate_data_for_batch()

        log('# sent: {}, sentence length: {}, vocab size: {}'.format(
            self.sentence_size, self.sentence_length, self.vocab_size))

    """preprocess data."""
    def preprocess(self):
        """preprocess the dataset."""
        log('preprocess the dataset.')

        log('load data.')
        if not exists(self.path_data_content):
            log('init content from raw.')
            sentence_token = self.init_content()

        log('...mask and pad the sentence.')
        padded_sentences_tokens, mask_sentences = self.mask_sentence(
            sentence_token)
        self.vocab, self.words = self.build_vocab(padded_sentences_tokens)

        log('...map word to index.')
        self.tensor = [
            ([self.vocab.get(s, self.vocab.get('unk'))
              for s in left_sentence],
             [self.vocab.get(s, self.vocab.get('unk'))
              for s in right_sentences])
            for left_sentence, right_sentences in padded_sentences_tokens]
        self.mask = mask_sentences

        log('...save processed data to file.')
        data = {
            'vocab': self.vocab,
            'words': self.words,
            'tensor': self.tensor,
            'mask': self.mask
        }
        opfile.write_cpickle(data, self.path_data_out_dict)

    def mapping(self, _inverse_dict, val_list):
        ret = [_inverse_dict[idx] for idx in val_list]
        return ['go'] + ret + ['eos']

    def make_square(self, seq, size):
        return zip(*[iter(seq)] * size)

    def read_txt(self, path, size):
        with open(path, 'r') as f:
            data = f.read()
        data = re.findall(r'\[(.*?)\]', data, re.S)
        data = self.make_square(data, size)
        data = [
            map(lambda x:
                map(int,
                    filter(
                        lambda y: y is not '',
                        x.replace(']', '').replace('[', '').strip().split(', '))
                    ),
                instance)
            for instance in data]

        print('----------------------')
        print('number of instances: {}'.format(len(data)))
        return data

    def init_content(self):
        """load data text.txt and put them line by line"""
        log('init data from the raw dataset.')
        log('...basic initialization')
        js_path = join(self.path_data_input, 'dict.json')
        ls_path = join(self.path_data_input, 'text.txt')

        log('...load the mapping dictionary.')
        _j = open(js_path).read().decode('utf-8')
        _dict = json.loads(_j)
        _inverse_dict = dict((v, k.encode('utf-8')) for k, v in _dict.items())

        log('...load data and do mapping.')
        ls = self.read_txt(ls_path, 2)
        sentences_tokens = []
        for l, r in ls:
            sentences_tokens.append(self.mapping(_inverse_dict, l))
            sentences_tokens.append(self.mapping(_inverse_dict, r))
        left_sentences_tokens = sentences_tokens[:-1]
        right_sentences_tokens = sentences_tokens[1:]
        return zip(left_sentences_tokens, right_sentences_tokens)

    def mask_sentence(self, sentences):
        """pad the sentence to a fixed length."""
        left_sentences = [val[0] for val in sentences]
        right_sentences = [val[1] for val in sentences]

        log('...statistics.')
        max_l_len, median_l_len, min_l_len = self.sentence_statistics(
            left_sentences, 'left sentence')
        max_r_len, median_r_len, min_r_len = self.sentence_statistics(
            right_sentences, 'right sentence')

        log('...build valid sentence and mask.')
        padded_l_sentences = [
            s + ['pad'] * (max_l_len - len(s)) for s in left_sentences]
        padded_r_sentences = [
            s + ['pad'] * (max_r_len - len(s)) for s in right_sentences]
        padded_sentences = zip(padded_l_sentences, padded_r_sentences)

        mask_l = [
            [1] * len(s) + [0] * (max_l_len - len(s)) for s in left_sentences]
        mask_r = [
            [1] * len(s) + [0] * (max_r_len - len(s)) for s in right_sentences]
        mask = zip(mask_l, mask_r)
        return padded_sentences, mask

    def sentence_statistics(self, sentences, name):
        sentence_lengths = map(lambda s: len(s), sentences)
        max_len = np.max(sentence_lengths)
        median_len = np.median(sentence_lengths)
        min_len = np.min(sentence_lengths)
        log('......{} stat: max len:{}, median len:{}, min len:{}'.format(
            name, max_len, median_len, min_len))
        return max_len, median_len, min_len

    def build_vocab(self, tokens):
        """build a vocabulary."""
        log('build a vocabulary.')
        log('...flatmap a list of sentence list to a list of sentence.')

        words = []
        for l, r in tokens:
            words += l + r
        word_counts = collections.Counter(words)

        log('...mapping from index to word.')
        word_list = [x[0] for x in word_counts.most_common()]

        log('...keep frequent words.')
        word_list = self.keep_common_words(word_list)

        log('...mapping from word to index.')
        vocab = {x: i for i, x in enumerate(word_list)}
        return vocab, word_list

    def keep_common_words(self, word_list):
        """keep common words."""
        if self.para.MAX_VOCAB_SIZE > 1000:
            word_list = word_list[: self.para.MAX_VOCAB_SIZE]
        word_list = ['unk'] + word_list
        return list(set(word_list))

    """for batch data generation."""
    def create_batches(self):
        """create batches for the dataset."""
        if not self.para.DEBUG:
            self.num_batches = int(len(self.tensor) / self.para.BATCH_SIZE)
        else:
            self.num_batches = int(self.para.DEBUG_SIZE / self.para.BATCH_SIZE)

        log('get data info.')
        self.vocab_size = len(self.words)
        self.sentence_size = len(self.tensor)
        self.sentence_length = len(self.tensor[0][0])

        num_samples = self.num_batches * self.para.BATCH_SIZE
        self.tensor = self.tensor[: num_samples]
        self.mask = self.mask[: num_samples]
        self.num_batches_train = int(self.num_batches * self.para.TRAIN_RATIO)
        self.num_batches_val = self.num_batches - self.num_batches_train

        log('init batch data.')
        log('...number of batches: {}'.format(self.num_batches))
        x = [(l, r[:-1]) for l, r in self.tensor]
        y = [r[1:] for l, r in self.tensor]
        ymask = [(l, r[1:]) for l, r in self.mask]
        z = np.random.uniform(
            size=(self.num_batches * self.para.BATCH_SIZE,
                  self.para.EMBEDDING_SIZE)
        )

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
