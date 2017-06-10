import numpy as np
import tensorflow as tf
import os
import json

class seq2seq(object):
    def __init__(self, output_size, hidden_size, projection_size,
                 embedding_size, batch_size=128, vocab_size=20525,
                 num_layers=1, keep_prob=0.95, truncated_std=0.1):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.truncated_std = truncated_std
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.initialize_placeholders()

    def initialize_placeholders(self):
        self.enc_inputs = tf.placeholder(tf.int32, shape=(None, self.batch_size), name="enc_inputs")
        self.targets = tf.placeholder(tf.int32, shape=(None, self.batch_size), name="targets")
        self.dec_inputs = tf.placeholder(tf.int32, shape=(None, self.batch_size), name="dec_inputs")
        self.emb_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=self.truncated_std), name="emb_weights")

        self.enc_inputs_emb = tf.nn.embedding_lookup(self.emb_weights, self.enc_inputs, name="enc_inputs_emb" )
        self.dec_inputs_emb = tf.nn.embedding_lookup(self.emb_weights, self.dec_inputs, name="dec_inputs_emb")
        self.initialize_input_layers()

    def initialize_input_layers(self):
        self.cell_list = []
        for i in xrange(self.num_layers):
             single_cell = tf.contrib.rnn.LSTMCell(
             num_units=self.hidden_size,
             num_proj=self.projection_size,
             state_is_tuple=True
             )
        # add dropout wrapper
        if i<self.num_layers-1 or self.num_layers ==1:
            single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell,
            output_keep_prob=self.keep_prob)

        self.cell_list.append(single_cell)
        self.cell = tf.contrib.rnn.MultiRNNCell(cells=self.cell_list, state_is_tuple=True)

    def _seq2seq(self):
        _, self.enc_states = tf.nn.dynamic_rnn(
        cell=self.cell,
        inputs=self.enc_inputs_emb,
        dtype=tf.float32,
        time_major=True,
        scope="encoder")

        self.dec_outputs, self.dec_states = tf.nn.dynamic_rnn(
        cell=self.cell,
        inputs=self.dec_inputs_emb,
        initial_state=self.enc_states,
        dtype=tf.float32,
        time_major=True,
        scope="decoder"
        )

        #output layers
        project_w = tf.Variable(tf.truncated_normal(shape=[self.output_size, self.embedding_size],
                          stddev=self.truncated_std), name="project_w")
        project_b = tf.Variable(tf.constant(shape=[self.embedding_size], value=0.1), name="project_b")

        softmax_w = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.vocab_size],
                          stddev=self.truncated_std), name="softmax_w")
        softmax_b = tf.Variable(tf.constant(shape=[self.vocab_size], value=0.1), name="softmax_b")

        self.dec_outputs = tf.reshape(self.dec_outputs, [-1, self.output_size], name="dec_outputs")
        dec_proj = tf.matmul(self.dec_outputs, project_w) + project_b
        logits = tf.nn.log_softmax(tf.matmul(dec_proj, softmax_w) + softmax_b, name="logits")
        probs = tf.nn.softmax(tf.matmul(dec_proj, softmax_w) + softmax_b, name="probs")
        #loss function
        flat_targets = tf.reshape(self.targets, [-1])
        self.total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets)
        self.avg_loss = tf.reduce_mean(self.total_loss)

        return self.total_loss, self.avg_loss, logits, self.enc_states, self.dec_outputs, self.dec_states, probs

    def get_emb(self, sess):
        return sess.run(self.emb_weights)
