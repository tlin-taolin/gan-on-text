# -*- coding: utf-8 -*-
import tensorflow as tf


class LSTM(object):
    def __init__(self, rnn_size, batch_size):
        self.rnn_size = rnn_size
        self.batch_size = batch_size

        self.define_recurrent_variables()

    def define_recurrent_variables(self):
        """define all variables that are used in the lstm."""
        # input gate.
        self.W_i = self.get_variable(shape=[self.rnn_size, 1], name='W_i')
        self.U_i = self.get_variable(shape=[self.rnn_size, 1], name='U_i')
        self.b_i = self.get_variable(shape=[self.rnn_size, 1], name='b_i')

        # forget gate.
        self.W_f = self.get_variable(shape=[self.rnn_size, 1], name='W_f')
        self.U_f = self.get_variable(shape=[self.rnn_size, 1], name='U_f')
        self.b_f = self.get_variable(shape=[self.rnn_size, 1], name='b_f')

        # output gate.
        self.W_o_g = self.get_variable(shape=[self.rnn_size, 1], name='W_o_g')
        self.U_o_g = self.get_variable(shape=[self.rnn_size, 1], name='U_o_g')
        self.b_o_g = self.get_variable(shape=[self.rnn_size, 1], name='b_o_g')

        # new memeory.
        self.W_c = self.get_variable(shape=[self.rnn_size, 1], name='W_c')
        self.U_c = self.get_variable(shape=[self.rnn_size, 1], name='U_c')
        self.b_c = self.get_variable(shape=[self.rnn_size, 1], name='b_c')

        # output unit.
        self.W_o = self.get_variable(shape=[self.rnn_size, 1], name='W_o')
        self.b_o = self.get_variable(shape=[self.rnn_size, 1], name='b_o')

    def get_variable(self, shape, initmethod=tf.truncated_normal,
                     name="W", trainable=True):
        """init weight."""
        initial = initmethod(shape, stddev=0.1)
        return tf.Variable(initial, name=name, trainable=trainable)

    def init_lstm_state(self):
        """init the lstm state."""
        h = tf.zeros([self.batch_size, 1])
        c = tf.zeros([self.batch_size, 1])
        return h, c

    def run_recurrent_unit(self, x_t, h_prev, c_prev):
        """create the recurrent unit."""
        # forget gate.
        f = tf.sigmoid(
            tf.matmul(x_t, self.W_f) +
            tf.matmul(h_prev, self.U_f) +
            self.b_f
        )

        # new memory cell.
        c_ = tf.nn.tanh(
            tf.matmul(x_t, self.W_c) +
            tf.matmul(h_prev, self.U_c) +
            self.b_c
        )

        # input gate.
        i = tf.sigmoid(
            tf.matmul(x_t, self.W_i) +
            tf.matmul(h_prev, self.U_i) +
            self.b_i
        )

        # new cell state.
        c = tf.multiply(f, c_prev) + tf.multiply(i, c_)

        # output gate.
        o = tf.sigmoid(
            tf.matmul(x_t, self.W_o_g) +
            tf.matmul(h_prev, self.U_o_g) +
            self.b_o_g
        )

        # hidden state.
        h = tf.multiply(o, tf.tanh(c))
        return h, c

    def recurrence_inference(self, time_step, x):
        """define the detailed the recurrence inference structure."""
        if time_step == 0:
            h, c = self.init_lstm_state()
            self.h, self.c = h, c
        self.h, self.c = self.run_recurrent_unit(x, self.h, self.c)
        return self.h
