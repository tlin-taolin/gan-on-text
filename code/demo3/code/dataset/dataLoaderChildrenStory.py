# -*- coding: utf-8 -*-
import re
from os.path import join

import numpy as np

import code.utils.opfiles as opfile
from code.utils.logger import log
from dataLoaderBasic import BasicLoader


class DataLoaderChildrenStory(BasicLoader):

    def __init__(self, para):
        """init parameters."""
        super(DataLoaderChildrenStory, self).__init__()

        self.para = para
        data_in_dir = para.RAW_DATA_DIRECTORY
        data_out_dir = para.DATA_DIRECTORY

        # define path and name.
        self.name = 'children_story'

        data_in_dir = join(data_in_dir, 'text', 'The_Childrens_Book_Test')
        self.path_data_input1 = join(data_in_dir, 'ROCStories_spring2016.csv')
        self.path_data_input2 = join(data_in_dir, 'ROCStories_winter2017.csv')
        self.path_data_inputs = [self.path_data_input1, self.path_data_input2]
        self.path_data_out_dir = join(data_out_dir, self.name)
        self.path_data_histogram = join(data_out_dir, 'histogram')
        self.path_data_out_dict = join(self.path_data_out_dir, 'data.pkl')
        self.path_data_out_string = join(self.path_data_out_dir, 'data.txt')

        # load the data, potentially need to do the preprocessing.
        self.load_data(para.REBUILD_DATA)
        log('the number of sentence is {}, the vocab size is {}'.format(
            self.sentence_size, self.vocab_size))

        # split the data or restore them from path.
        self.generate_data_for_batch()

    def preprocess(self):
        """preprocess the dataset."""
        log('preprocess the dataset.')
        sentences_string = []
        sentences_tokens = []

        for path_data_input in self.path_data_inputs:
            with open(path_data_input, "r") as f:
                data = f.read()

            sentence_string, sentence_token = self.advance_cleaner(data)
            sentences_string += sentence_string
            sentences_tokens += sentence_token

        opfile.output_string(sentences_string, self.path_data_out_string)

        log('...mask and pad the sentence.')
        padded_sentences_tokens, mask_sentences = self.mask_sentence(
            sentences_tokens)

        self.vocab, self.words = self.build_vocab(padded_sentences_tokens)
        self.sentences_tokens = padded_sentences_tokens

        log('...map word to index.')
        self.tensor = np.array([
            [self.vocab.get(s) for s in sentence]
            for sentence in padded_sentences_tokens])
        self.mask = np.array(mask_sentences)

        log('...some data statistics.')
        self.sentence_size = self.tensor.shape[0]
        self.sentence_length = self.tensor.shape[1]
        self.vocab_size = len(self.words)

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
        upper = 3
        lower = 5

        log('......max len:{}, median len:{}, min len:{}'.format(
            max_len, median_len, min_len))
        valid_sentences = filter(
            lambda s: len(s) >= lower and len(s) <= max_len - upper, sentences)
        padded_sentences = [
            s + ['<pad>'] * (max_len - len(s))
            for s in valid_sentences]
        mask = [
            [1] * len(s) + [0] * (max_len - len(s))
            for s in valid_sentences]
        return padded_sentences, mask

    def advance_cleaner(self, string):
        """advance cleaner, normally specifically for this dataset."""
        string = self.basic_cleaner(string)

        string = string.split('\r')[1:]
        string = [s.split(',')[2: 3] + s.split('. ,')[3:] for s in string]
        sentences = [
            '<go> '+' . '.join([self.cleaner_assist(x)+' <eos>' for x in s])
            for s in string]
        token = [sentence.split(' ') for sentence in sentences]
        return sentences, token

    def cleaner_assist(self, x):
        """an assistant to do the cleaning."""
        x = re.sub(r'[:|\-|/|+]', r' ', x)
        x = re.sub(r'[\d*|`|\%\%|\&|\%]', r'', x)
        x = re.sub(r',', '', x)
        x = re.sub(r'\.', '', x)
        x = re.sub(r"\s{2,}", " ", x)
        return x.strip()
