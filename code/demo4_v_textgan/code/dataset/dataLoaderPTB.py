# -*- coding: utf-8 -*-
import re
import collections
from os.path import exists, join
from os import listdir
import numpy as np

from code.utils.logger import log
import code.utils.opfiles as opfile
import code.dataset.parseEmbedding as parse_embedding
from code.dataset.dataLoaderBasic import DataLoaderBasic


class DataLoaderPTB(DataLoaderBasic):

    def __init__(self, para):
        """init parameters."""
        super(DataLoaderPTB, self).__init__()

        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'ptb'

        self.path_data_input = join(data_in_dir, 'text', 'PTB')
        self.path_data_content = join(self.path_data_input, 'all')
        self.path_data_out_dir = join(data_out_dir, self.name)
        self.path_data_out_dict = join(self.path_data_out_dir, 'data.pkl')
        self.path_data_out_string_tr = join(
            self.path_data_out_dir, 'data_tr.txt')
        self.path_data_out_string_val = join(
            self.path_data_out_dir, 'data_val.txt')

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

        path_ptb_train = join(self.path_data_input, 'train.txt')
        path_ptb_val = join(self.path_data_input, 'eval.txt')

        ptb_train = opfile.read_text_withoutsplit(path_ptb_train).replace(
            '\n', ' ').replace('<START>', '\n').replace('<END>', '\n')
        ptb_val = opfile.read_text_withoutsplit(path_ptb_val).replace(
            '\n', ' ').replace('<START>', '\n').replace('<END>', '\n')

        log('load context for further preprocessing.')
        sentence_string_tr, sentence_token_tr = self.advance_cleaner(ptb_train)
        sentence_string_val, sentence_token_val = self.advance_cleaner(ptb_val)

        log('...mask and pad the sentence.')
        padded_sentences_tokens_tr, mask_sentences_tr \
            = self.mask_sentence(sentence_token_tr)
        padded_sentences_tokens_val, mask_sentences_val \
            = self.mask_sentence(sentence_token_val)

        padded_sentences_tokens \
            = padded_sentences_tokens_tr + padded_sentences_tokens_val
        mask_sentences = mask_sentences_tr + mask_sentences_val
        self.vocab, self.words = self.build_vocab(padded_sentences_tokens)

        opfile.write_txt(sentence_string_tr, self.path_data_out_string_tr)
        opfile.write_txt(sentence_string_val, self.path_data_out_string_val)

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

        log('...load existing embedding')
        word_embedding_dict = parse_embedding.load_word_embedding(
            self.para.PRE_EMBEDDING_DIRECTORY,
            self.para.EMBEDDING_SIZE)
        embedding, vocab = parse_embedding.check_and_replace_nonexisting(
            word_embedding_dict, vocab)
        self.embedding_matrix = embedding
        return vocab, word_list

    def advance_cleaner(self, string):
        """advance cleaner, normally specifically for this dataset."""
        log('clean data.')

        string = string.strip().lower()

        if self.para.CLEAN_DATA:
            string = self.basic_cleaner(string)

        sentences = string.split(' . ')
        sentences = [s.split() for s in sentences]

        sentences_tokens = self.magicsplit(sentences, '\n')[0]
        sentences_tokens = [
            ['go'] + filter(lambda x: x != '', s) + ['eos']
            for s in sentences]

        string = [' '.join(s) for s in sentences_tokens]
        string = '\n'.join(string)
        return string, sentences_tokens

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
                  self.para.EMBEDDING_SIZE * 2))

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

    """some utilities."""
    def magicsplit(self, l, *splitters):
        def _itersplit(l, splitters):
            current = []
            for item in l:
                if item in splitters:
                    yield current
                    current = []
                else:
                    current.append(item)
            yield current
        return [subl for subl in _itersplit(l, splitters) if subl]
