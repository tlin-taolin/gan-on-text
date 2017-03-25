# -*- coding: utf-8 -*-
import sys
import numpy as np


class WordSearch(object):

    def __init__(self, para, vocab):
        self.beam_candidates = []
        self.beam_options = []
        self.best_sequence = []
        self.highest_score = - sys.maxint - 1
        self.vocab = vocab
        self.para = para

    def weighted_pick(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return int(np.searchsorted(t, np.random.rand(1) * s))

    def beam_search(
            self, sess, placeholder_x, placeholder_dropout,
            placeholder_g_state, top_value, top_index, state):
        for i in range(len(self.beam_candidates)):
            cur_word_score = self.beam_candidates[i][0]
            cur_word_indices = self.beam_candidates[i][1]

            if len(cur_word_indices) > self.para.SENTENCE_LENGTH_TO_GENERATE:
                if cur_word_score > self.highest_score:
                    self.highest_score = cur_word_score
                    self.best_sequence = cur_word_indices
                continue

            values, indices = sess.run(
                [top_value, top_index],
                {placeholder_x: [cur_word_indices[-1:]],
                 placeholder_dropout: 1.0, placeholder_g_state: state})

            values, indices = values[0][0], indices[0][0]

            for j in range(len(values)):
                new_seq = list(cur_word_indices)
                new_seq.append(indices[j])
                self.beam_options.append([cur_word_score + values[j], new_seq])

        self.beam_options.sort(reverse=True)
        self.beam_candidates = []

        for i in range(min(len(self.beam_options), self.para.BEAM_SEARCH_SIZE)):
            if self.beam_options[i][0] > self.highest_score:
                self.beam_candidates.append(self.beam_options[i])

        self.beam_options = []
