# -*- coding: utf-8 -*-
import tensorflow as tf


class LSTM(object):
    def __init__(self, rnn_size, batch_size):
        self.rnn_size = rnn_size
        self.batch_size = batch_size

        self.define_recurrent_variables()

    def inherit_lstm_fn_from_tf(self, para, model_type):
        if model_type == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif model_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif model_type == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(
                para.MODEL_TYPE))

        # define cell architecture.
        cell = cell_fn(para.RNN_SIZE, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=para.DROPOUT_RATE)
        cell = tf.contrib.rnn.MultiRNNCell(
            [cell] * para.RNN_DEPTH, state_is_tuple=True)
        return cell

    def define_recurrent_variables(self):
        """define all variables that are used in the lstm."""
        # input gate.
        self.W_i = self.get_variable(shape=[self.rnn_size, 1], name='W_i')
        self.U_i = self.get_variable(shape=[self.rnn_size, 1], name='U_i')
        self.C_i = self.get_variable(shape=[self.rnn_size, 1], name='C_i')
        self.b_i = self.get_variable(shape=[self.rnn_size], name='b_i')

        # forget gate.
        self.W_f = self.get_variable(shape=[self.rnn_size, 1], name='W_f')
        self.U_f = self.get_variable(shape=[self.rnn_size, 1], name='U_f')
        self.C_f = self.get_variable(shape=[self.rnn_size, 1], name='C_f')
        self.b_f = self.get_variable(shape=[self.rnn_size], name='b_f')

        # output gate.
        self.W_o_g = self.get_variable(shape=[self.rnn_size, 1], name='W_o_g')
        self.U_o_g = self.get_variable(shape=[self.rnn_size, 1], name='U_o_g')
        self.C_o_g = self.get_variable(shape=[self.rnn_size, 1], name='C_o_g')
        self.b_o_g = self.get_variable(shape=[self.rnn_size], name='b_o_g')

        # new memeory.
        self.W_c = self.get_variable(shape=[self.rnn_size, 1], name='W_c')
        self.U_c = self.get_variable(shape=[self.rnn_size, 1], name='U_c')
        self.C_c = self.get_variable(shape=[self.rnn_size, 1], name='C_c')
        self.b_c = self.get_variable(shape=[self.rnn_size], name='b_c')

        # output unit.
        self.W_o = self.get_variable(shape=[self.rnn_size, 1], name='W_o')
        self.b_o = self.get_variable(shape=[self.rnn_size], name='b_o')

    def get_variable(self, shape, initmethod=tf.truncated_normal,
                     name="W", trainable=True):
        """init weight."""
        initial = initmethod(shape, stddev=0.1)
        return tf.Variable(initial, name=name, trainable=trainable)

    def init_state(self):
        """init the lstm state."""
        c = tf.zeros([self.batch_size, self.rnn_size], tf.float32)
        h = tf.zeros([self.batch_size, self.rnn_size], tf.float32)
        state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
        return state

    def standard_lstm_unit(self, x_t, state, z_t=None):
        """create the recurrent unit."""
        # forget gate.
        c, h = state.c, state.h
        f = tf.sigmoid(
            tf.matmul(x_t, self.W_f) +
            tf.matmul(h, self.U_f) +
            self.b_f
        )

        # new memory cell.
        c_ = tf.nn.tanh(
            tf.matmul(x_t, self.W_c) +
            tf.matmul(c, self.U_c) +
            self.b_c
        )

        # input gate.
        i = tf.sigmoid(
            tf.matmul(x_t, self.W_i) +
            tf.matmul(h, self.U_i) +
            self.b_i
        )

        # new cell state.
        c = tf.multiply(f, c) + tf.multiply(i, c_)

        # output gate.
        o = tf.sigmoid(
            tf.matmul(x_t, self.W_o_g) +
            tf.matmul(h, self.U_o_g) +
            self.b_o_g
        )

        # hidden state.
        h = tf.multiply(o, tf.tanh(c))

        # convert to state.
        state = tf.contrib.rnn.LSTMStateTuple(c, h)
        return state

    def noise_lstm_unit(self, x_t, state, z_t):
        """create the recurrent unit."""
        # forget gate.
        c, h = state.c, state.h
        f = tf.sigmoid(
            tf.matmul(x_t, self.W_f) +
            tf.matmul(h, self.U_f) +
            tf.matmul(z_t, self.C_f) +
            self.b_f
        )

        # new memory cell.
        c_ = tf.nn.tanh(
            tf.matmul(x_t, self.W_c) +
            tf.matmul(c, self.U_c) +
            tf.matmul(z_t, self.C_c) +
            self.b_c
        )

        # input gate.
        i = tf.sigmoid(
            tf.matmul(x_t, self.W_i) +
            tf.matmul(h, self.U_i) +
            tf.matmul(z_t, self.C_i) +
            self.b_i
        )

        # new cell state.
        c = tf.multiply(f, c) + tf.multiply(i, c_)

        # output gate.
        o = tf.sigmoid(
            tf.matmul(x_t, self.W_o_g) +
            tf.matmul(h, self.U_o_g) +
            self.b_o_g
        )

        # hidden state.
        h = tf.multiply(o, tf.tanh(c))

        # convert to state.
        state = tf.contrib.rnn.LSTMStateTuple(c, h)
        return state

    def cell(self, cell_type, x, state, z=None):
        """define the detailed the recurrence inference structure."""
        state = cell_type(x, state, z)
        return state.h, state
