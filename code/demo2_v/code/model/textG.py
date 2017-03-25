# -*- coding: utf-8 -*-
import tensorflow as tf

from code.core.lstm import LSTM
from code.core.basicModel import BasicModel


class TextG(BasicModel):
    """The textG here simply follow the paper.
    It uses an RNN network as the generator.
    """

    def __init__(self, para, loader, sess, training=True):
        """init parameters."""
        super(TextG, self).__init__(para, loader, sess, training)

        # init the basic model..
        self.define_placeholder()
        self.lstm = LSTM(para.RNN_SIZE, para.BATCH_SIZE)

        self.G_cell = self.lstm.inherit_lstm_fn_from_tf(para, 'lstm')
        self.G_cell_init_state = self.G_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)

    def define_generator(self):
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        inputs = self.embedding(self.x, reuse=True)
        input = inputs[:, 0, :]

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0:
                    scope_rnn.reuse_variables()
                cell_output, state = self.G_cell(input, state)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            input = inputs[:, time_step, :]

            # save the middle result.
            logits.append(logit)
            probs.append(prob)
            outputs.append(output)
            embedded_outputs.append(input)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        embedded_outputs = tf.reshape(
            tf.concat(embedded_outputs, 0),
            [-1, self.loader.sentence_length, self.para.EMBEDDING_SIZE])
        return logits, probs, outputs, embedded_outputs, state

    def inference(self):
        """"define the inference procedure in training phase."""
        with tf.variable_scope('generator'):
            self.embedding()
            self.language_model()
            self.logits, self.probs, self.outputs,\
                _, self.final_state = self.define_generator()

            self.top_value, self.top_index = tf.nn.top_k(
                self.logits, k=self.para.BEAM_SEARCH_SIZE, sorted=True)

    def define_loss(self):
        """define the loss."""
        with tf.name_scope("loss"):
            self.loss_G = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)
