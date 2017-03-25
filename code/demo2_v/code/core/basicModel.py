# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import sys
import time
import random
import datetime
from os.path import join, exists

import numpy as np
import tensorflow as tf

import code.utils.auxiliary as auxi
import code.utils.opfiles as opfile
from code.utils.logger import log
from code.core.wordSearch import WordSearch


class BasicModel(object):
    """a base model for any subsequent implementation."""
    def __init__(self, para, loader, sess, training):
        """init."""
        # system define.
        self.para = para
        self.loader = loader
        self.training = training
        self.sess = sess

        if not training:
            self.para.BATCH_SIZE = 1
            self.loader.sentence_length = 1

    """define placeholder."""
    # define the basic components of the inference procedure.
    def define_placeholder(self):
        """define the placeholders."""
        self.x = tf.placeholder(
            tf.int32,
            [None, self.loader.sentence_length], name="x")
        self.y = tf.placeholder(
            tf.int32,
            [None, self.loader.sentence_length], name="y")
        self.ymask = tf.placeholder(
            tf.float32,
            [None, self.loader.sentence_length], name="ymask")
        self.dropout_val = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    """training/optimizer related."""
    # define optimiaer for training
    def define_optimizer(self, learning_rate):
        if self.para.OPTIMIZER_NAME == 'Adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.para.OPTIMIZER_NAME == 'RMSProp':
            return tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise 'not a vaild optimizer.'

    # define pretrain related
    def define_train_op(self):
        # get the training vars for both networks
        vars_all = tf.trainable_variables()
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # define optimizer
        optimizer_G = self.define_optimizer(self.para.LEARNING_RATE_G)

        grads_G, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_G, vars_G), self.para.GRAD_CLIP)
        self.grads_and_vars_G = zip(grads_G, vars_G)
        self.op_train_G = optimizer_G.apply_gradients(self.grads_and_vars_G)

    def train_step(self, losses):
        """pretrain the model."""
        batch_x, batch_y, batch_ymask = self.loader.next_batch()

        feed_dict_G = {
            self.x: batch_x, self.y: batch_y, self.ymask: batch_ymask,
            self.dropout_val: self.para.DROPOUT_RATE}

        # train G.
        _, loss_G, summary_G = self.sess.run(
            [self.op_train_G, self.loss_G, self.train_summary_G_op],
            feed_dict=feed_dict_G)

        # record loss.
        losses['losses_G'].append(loss_G)

        # summary
        self.train_summary_writer.add_summary(summary_G)
        return losses

    def val_step(self, losses):
        """pretrain the model."""
        batch_x, batch_y, batch_ymask = self.loader.next_batch()

        feed_dict_G = {
            self.x: batch_x, self.y: batch_y, self.ymask: batch_ymask,
            self.dropout_val: 1.0}

        # train G.
        loss_G, summary_G = self.sess.run(
            [self.loss_G, self.train_summary_G_op], feed_dict=feed_dict_G)

        # record loss.
        losses['losses_G'].append(loss_G)

        # summary
        self.val_summary_writer.add_summary(summary_G)
        return losses

    def run_epoch(self, stage):
        """run pretrain epoch."""
        losses = {'losses_G': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()

        batch_scope = self.loader.determine_batch_pointer_pos(stage)
        for step in batch_scope:
            losses = stage(losses)

            sys.stdout.write(
                '\r{}/{}: train: mean loss G = {}'.format(
                    step, batch_scope[-1], np.mean(losses['losses_G']))
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = 1.0 * (
            end_epoch_time - start_epoch_time).seconds/self.loader.num_batches
        return np.mean(losses['losses_G']), duration

    """define status tracking stuff."""
    # tracking status.
    def define_keep_tracking(self):
        self.build_dir_for_tracking()
        self.keep_tracking_train()

    def build_dir_for_tracking(self):
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        datatype = self.loader.__class__.__name__
        parent_path = join(self.para.TRAINING_DIRECTORY, "runs", datatype)
        method_folder = join(parent_path, auxi.get_fullname(self))
        self.out_dir = join(method_folder, timestamp)

        # Checkpoint directory. Tensorflow assumes this directory
        # already exists so we need to create it
        checkpoint_dir = join(self.out_dir, "checkpoints")
        self.checkpoint_prefix = join(checkpoint_dir, "model")
        self.checkpoint_comparison = join(checkpoint_dir, "comparison")
        self.best_model = join(checkpoint_dir, "bestmodel")
        if not exists(checkpoint_dir):
            opfile.build_dirs(self.checkpoint_comparison)
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.para.MAX_MODEL_TO_KEEP)

    def keep_tracking_train(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_G)

        # Summaries for loss and accuracy
        loss_summary_G = tf.summary.scalar("loss_G", self.loss_G)

        # Train Summaries
        self.train_summary_G_op = tf.summary.merge(
            [loss_summary_G, grad_summaries_merged_G])

        log("writing to {}\n".format(self.out_dir))
        train_summary_dir = join(self.out_dir, "summaries", "train")
        val_summary_dir = join(self.out_dir, "summaries", "val")

        self.train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(
            val_summary_dir, self.sess.graph)

    def keep_tracking_grad_and_vals(self, grads_and_vars):
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        return grad_summaries_merged

    def define_variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    """define some common operations that are involved in inference."""
    # define some common operation.
    def conv2d(self, x, W, strides, padding='SAME', name="conv"):
        """do convolution.
            x: [batch, in_height, in_width, in_channels]
            W: [filter_height, filter_width, in_channels, out_channels]
        """
        return tf.nn.conv2d(
            x, W,
            strides=strides,
            padding=padding, name=name)

    def avg_pool(self, x, ksize, strides, padding='SAME', name="pool"):
        """do average pooling."""
        return tf.nn.avg_pool(
            x, ksize=ksize, strides=strides, padding=padding, name=name)

    def max_pool(self, x, ksize, strides, padding='SAME', name="pool"):
        """do max pooling.
            x: [batch, height, width, channels]
        """
        return tf.nn.max_pool(
            x, ksize=ksize, strides=strides, padding=padding, name=name)

    def weight_variable(
            self, shape, initmethod=tf.truncated_normal,
            name="W", trainable=True):
        """init weight."""
        initial = initmethod(shape, stddev=0.1)
        return tf.Variable(initial, name=name, trainable=trainable)

    def bias_variable(self, shape, name="b"):
        """init bias variable."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def leakyrelu_s(self, x, alpha=1/5.5):
        return tf.maximum(x, alpha * x)

    def leakyrelu(self, conv, b, alpha=0.01, name="leaky_relu"):
        """use lrelu as the activation function."""
        tmp = tf.nn.bias_add(conv, b)
        return tf.maximum(tmp * alpha, tmp)

    def tanh(self, conv, b, name="tanh"):
        """use tanh as the activation function."""
        return tf.tanh(tf.nn.bias_add(conv, b), name=name)

    """define method related to language model/word embedding."""
    def language_model(self, output=None, reuse=False):
        with tf.variable_scope('lm') as scope:
            if not reuse:
                softmax_w = tf.get_variable(
                    "softmax_w",
                    shape=[self.para.RNN_SIZE, self.loader.vocab_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                softmax_b = tf.get_variable(
                    "softmax_b",
                    shape=[self.loader.vocab_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.define_variable_summaries(softmax_w)
                self.define_variable_summaries(softmax_b)
            elif reuse and output is not None:
                scope.reuse_variables()
                softmax_w = tf.get_variable("softmax_w")
                softmax_b = tf.get_variable("softmax_b")

                logit = tf.matmul(output, softmax_w) + softmax_b
                prob = tf.nn.softmax(logit)
                soft_prob = tf.nn.softmax(logit * self.para.L_SOFT)
                output = tf.stop_gradient(tf.argmax(prob, 1))
                return logit, prob, soft_prob, output
            else:
                raise 'invaild usage.'

    def embedding(self, words=None, reuse=False):
        """word embedding."""
        with tf.variable_scope("embedding") as scope:
            if not reuse:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        "embedding",
                        shape=[
                            self.loader.vocab_size, self.para.EMBEDDING_SIZE],
                        initializer=tf.contrib.layers.xavier_initializer())
            else:
                scope.reuse_variables()
                embedding = tf.get_variable("embedding")
                embedded_words = tf.nn.embedding_lookup(embedding, words)
                return embedded_words

    def get_approx_embedding(self, soft_prob):
        with tf.variable_scope("embedding") as scope:
            scope.reuse_variables()
            embedding = tf.get_variable("embedding")
        return tf.matmul(soft_prob, embedding)

    # add function to generate sentence.
    def sample_from_latent_space(self, vocab_word2index, vocab_index2word):
        """generate sentence from latent space.."""
        # init.
        state = self.sess.run(self.G_cell.zero_state(1, tf.float32))
        input = [[random.randint(0, len(vocab_word2index))]]

        # some preparation.
        # placeholders.
        generated_sentence = []
        word_search = WordSearch(self.para, self.loader.vocab)

        # generate first word
        log('sample from the continuous space.')
        log('...generate the first word.')
        probs, state, values, indices = self.sess.run(
            [self.probs, self.final_state, self.top_value, self.top_index],
            {self.x: input, self.dropout_val: 1.0,
             self.G_cell_init_state: state})

        # prepare basic sampler.
        if self.para.SAMPLING_TYPE == 'argmax':
            basic_sampler = np.argmax
        else:
            basic_sampler = word_search.weighted_pick
        sample_index = basic_sampler(probs[0, 0])
        word = vocab_index2word[sample_index]
        generated_sentence.append(word)
        input = [[sample_index]]

        # prepare beam sampler.
        values, indices = values[0][0], indices[0][0]
        for i in range(len(values)):
            word_search.beam_candidates.append((values[i], [indices[i]]))

        log('...generate other words.')
        while True:
            if not self.para.BEAM_SEARCH:
                probs, state = self.sess.run(
                    [self.probs, self.final_state],
                    {self.x: input, self.dropout_val: 1.0,
                     self.G_cell_init_state: state})

                sample_index = basic_sampler(probs[0, 0])
                word = vocab_index2word[sample_index]
                generated_sentence.append(word)
                input = [[sample_index]]
            else:
                word_search.beam_search(
                    self.sess, self.x, self.dropout_val,
                    self.G_cell_init_state,
                    self.top_value, self.top_index, state)
                generated_sentence = [
                    vocab_index2word[s] for s in word_search.best_sequence]

            if len(generated_sentence) > self.para.SENTENCE_LENGTH_TO_GENERATE:
                break
        return generated_sentence
