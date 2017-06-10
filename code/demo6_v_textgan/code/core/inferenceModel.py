# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf

from code.utils.logger import log
from code.core.lstm import LSTM
from code.core.basicModel import BasicModel
from code.core.wordSearch import WordSearch


class InferenceModel(BasicModel):
    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(InferenceModel, self).__init__(para, loader, sess, infer)

        # init the basic model..
        self.define_placeholder()
        self.define_pointer()
        self.lstm = LSTM

    def do_gradient_penalty(self, real_inputs, fake_inputs):
        # WGAN lipschitz-penalty.
        epsilon = tf.random_uniform(
            shape=[self.para.BATCH_SIZE, 1, 1],
            minval=0.,
            maxval=1.
        )
        interpolates = real_inputs + epsilon * (fake_inputs - real_inputs)
        interpolates_logits, _ = self.define_discriminator(
            interpolates, reuse=True)

        interpolates_gradients = tf.gradients(
            interpolates_logits, [interpolates])[0]
        interpolates_slopes = tf.sqrt(
            tf.reduce_sum(tf.square(interpolates_gradients),
                          reduction_indices=[1, 2])
        )
        gradient_penalty = tf.reduce_mean((interpolates_slopes - 1.) ** 2)
        return tf.real(gradient_penalty)

    """define method related to language model/word embedding."""
    def projection(self, input=None):
        W = self.get_scope_variable(
            'projection', 'proj_W',
            shape=[self.para.PROJECTION_SIZE, self.para.EMBEDDING_SIZE])
        b = self.get_scope_variable(
            'projection', 'proj_b',
            shape=[self.para.EMBEDDING_SIZE])

        if input is not None:
            projection = tf.nn.xw_plus_b(input, W, b)
            return projection

    def language_model(self, input=None):
        softmax_w = self.get_scope_variable(
            'lm', 'softmax_w',
            shape=[self.para.PROJECTION_SIZE, self.loader.vocab_size])
        softmax_b = self.get_scope_variable(
            'lm', 'softmax_b',
            shape=[self.loader.vocab_size])

        if input is not None:
            input = tf.reshape(input, [-1, self.para.PROJECTION_SIZE])
            logit = tf.matmul(input, softmax_w) + softmax_b
            prob = tf.nn.softmax(logit)
            log_prob = tf.nn.log_softmax(logit)
            soft_prob = tf.nn.softmax(logit * self.soft_argmax)
            # output = tf.stop_gradient(tf.argmax(prob, 1))
            return logit, prob, log_prob, soft_prob

    def embedding_model(self, words=None, trainable=True):
        """word embedding."""
        with tf.device("/cpu:0"):
            embedding = self.get_scope_variable(
                'embedding', 'embedding_matrix',
                shape=[self.loader.vocab_size, self.para.EMBEDDING_SIZE],
                trainable=trainable)
            if words is not None:
                if words.dtype != tf.int32:
                    words = tf.cast(words, tf.int32)
                embedded_words = tf.nn.embedding_lookup(embedding, words)
                return embedded_words

    def get_approx_embedding(self, soft_prob, trainable=True):
        embedding = self.get_scope_variable(
            'embedding', 'embedding_matrix', trainable=trainable)
        return tf.matmul(soft_prob, embedding)

    def adjust_soft_argmax(self, cur_epoch):
        """adjust soft_argmax parameter.
            if self.para.EPOCH_TRAIN > self.para.SOFT_ARGMAX_UPPER_EPOCH:
                it will stop at the epoch of SOFT_ARGMAX_UPPER_EPOCH.
        """
        min_epoch = self.para.EPOCH_PRETRAIN
        duration_epoch = min(self.para.EPOCH_TRAIN,
                             self.para.SOFT_ARGMAX_UPPER_EPOCH)
        soft_argmax_range = map(float, self.para.SOFT_ARGMAX.split(','))
        soft_argmax_lb, soft_argmax_ub = soft_argmax_range
        soft_argmax_step = (soft_argmax_ub - soft_argmax_lb) / duration_epoch

        cur_soft_argmax = \
            soft_argmax_lb + (cur_epoch - min_epoch) * soft_argmax_step
        cur_soft_argmax = cur_soft_argmax \
            if cur_soft_argmax >= soft_argmax_lb else soft_argmax_lb
        return cur_soft_argmax

    def build_sample_input(self, sequence):
        dec_inp = np.zeros((1, len(sequence)))
        dec_inp[0][:] = sequence
        return dec_inp
