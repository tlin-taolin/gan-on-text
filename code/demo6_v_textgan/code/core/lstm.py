# -*- coding: utf-8 -*-

import tensorflow as tf


from code.utils.logger import log


class LSTM(object):
    def __init__(self, para):
        self.para = para

    def build_stacked_rnn(self, cell_fn, rnn_size, proj_size, keep_prob):
        """build stacked rnn."""
        cells = []
        for i in range(self.para.RNN_LAYER):
            single_cell = cell_fn(
                num_units=rnn_size,
                num_proj=proj_size,
                state_is_tuple=True)
            if i < self.para.RNN_LAYER - 1 or self.para.RNN_LAYER == 1:
                single_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=single_cell, output_keep_prob=keep_prob)
            cells.append(single_cell)
        return tf.contrib.rnn.MultiRNNCell(cells=cells, state_is_tuple=True)

    def inherit_lstm_fn_from_tf(
            self, rnn_type, rnn_size, proj_size, keep_prob=1.0):
        log('rnn type is {}'.format(rnn_type))

        if rnn_type == 'rnn':
            cell_fn = tf.contrib.rnn.RNNCell
        elif rnn_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif rnn_type == 'lstm':
            cell_fn = tf.contrib.rnn.LSTMCell
        else:
            raise Exception("model type not supported: {}".format(
                self.para.MODEL_TYPE))

        return self.build_stacked_rnn(cell_fn, rnn_size, proj_size, keep_prob)
