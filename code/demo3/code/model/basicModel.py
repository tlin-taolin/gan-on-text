# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import os
import sys
import time
import datetime
from os.path import join, exists

import numpy as np
import tensorflow as tf

import code.utils.auxiliary as auxi
from code.utils.logger import log
from code.model.beamSearch import BeamSearch


class BasicModel(object):
    """a base model for any subsequent implementation."""

    def __init__(self, para, loader, training):
        """init."""
        # system define.
        self.para = para
        self.loader = loader
        self.training = training

        if not training:
            self.para.BATCH_SIZE = 1
            self.loader.sentence_length = 1

    """define placeholder."""
    # define the basic components of the inference procedure.
    def define_placeholder(self):
        """define the placeholders."""
        self.z = tf.placeholder(
            tf.float32,
            [None, self.para.Z_DIM], name="z")
        self.x = tf.placeholder(
            tf.int32,
            [None, self.loader.sentence_length], name="x")
        self.x_label = tf.placeholder(
            tf.float32,
            [None, 1], name="x_label")
        self.y = tf.placeholder(
            tf.int32,
            [None, self.loader.sentence_length], name="y")
        self.ymask = tf.placeholder(
            tf.float32,
            [None, self.loader.sentence_length], name="ymask")
        self.dropout_val = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    """define inference main entry."""

    def define_inference(self, generator, discriminator):
        """"define the inference procedure in training phase."""
        with tf.variable_scope('generator'):
            self.embedding()
            self.language_model()

            self.yhat_logit, self.yhat_prob, self.yhat_out, _, self.yhat_state\
                = generator(x=self.x, pretrain=True)
            self.G_logit, self.G_prob, self.G_out, self.G_embedded_out, self.G_state \
                = generator(z=self.z, pretrain=False)
            embedded_x = self.embedding(self.x, reuse=True)

        if self.training:
            with tf.variable_scope('discriminator') as discriminator_scope:
                self.D_logit_real, self.D_real \
                    = discriminator(embedded_x, discriminator_scope)

                # get discriminator on fake data. the reuse=True, which
                # specifies we reuse the discriminator ops for new placeholder.
                self.D_logit_fake, self.D_fake \
                    = discriminator(self.G_embedded_out, discriminator_scope,
                                    reuse=True)

    """define inference components."""
    def define_discriminator_as_CNN(self, embedded, scope, reuse=False):
        """define the discriminator."""
        if reuse:
            scope.reuse_variables()

        input = tf.expand_dims(embedded, 3)

        # build a series for 'CONV + RELU + POOL' architecture.
        archits = zip(self.para.D_CONV_SPATIALS, self.para.D_CONV_DEPTHS)
        pooled_outputs = []
        for i, (conv_spatial, conv_depth) in enumerate(archits):
            with tf.variable_scope("conv-lrelu-pooling-%s" % i):
                W = self.weight_variable(
                    shape=[
                        conv_spatial,
                        self.para.EMBEDDING_SIZE,
                        1,
                        conv_depth])
                b = self.bias_variable(shape=[conv_depth])

                conv = self.conv2d(
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                h = self.leakyrelu(conv, b, alpha=1.0/5.5)
                pooled = self.max_pool(
                    h,
                    ksize=[
                        1,
                        self.loader.sentence_length - conv_spatial + 1,
                        1,
                        1],
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                pooled_outputs.append(pooled)

        num_filters_total = sum([x[1] for x in archits])
        h_pool_flat = tf.reshape(
            tf.concat(pooled_outputs, 3), [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_val)

        # output the classification result.
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.para.LABEL_DIM],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                'b',
                shape=[self.para.LABEL_DIM],
                initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        return logits, tf.nn.sigmoid(logits)

    def define_discriminator_as_LSTM(self, embedded, dis_scope, reuse=False):
        """define the discriminator."""
        state = self.D_cell_init_state

        with tf.variable_scope('rnn') as scope_rnn:
            for time_step in range(self.loader.sentence_length):
                if time_step > 0 or reuse:
                    scope_rnn.reuse_variables()
                output, state = self.D_cell(embedded[:, time_step, :], state)

        # output the classification result.
        with tf.variable_scope("output") as scope:
            if not reuse:
                W = tf.get_variable(
                    "W",
                    shape=[self.para.RNN_SIZE, self.para.LABEL_DIM],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(
                    'b',
                    shape=[self.para.LABEL_DIM],
                    initializer=tf.contrib.layers.xavier_initializer())
            else:
                scope.reuse_variables()
                W = tf.get_variable("W")
                b = tf.get_variable("b")
            logits = tf.nn.xw_plus_b(output, W, b)
        return logits, tf.nn.sigmoid(logits)

    def define_generator_as_LSTM(self, z=None, x=None, pretrain=False):
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            input = z

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()
                cell_output, state = self.G_cell(input, state)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = self.get_approx_embedding(soft_prob)

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

    def define_generator_as_noiseLSTM(self, z=None, x=None, pretrain=False):
        """define the generator.

        feed z to every lstm cells to guide the sentence generation.
        """
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            cell_type = self.lstm.standard_lstm_unit
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            cell_type = self.lstm.noise_lstm_unit
            input = z

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()
                if time_step == 0:
                    cell_output, state = self.lstm.cell(
                        self.lstm.standard_lstm_unit, input, state, z)
                else:
                    cell_output, state = self.lstm.cell(
                        cell_type, input, state, z)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = self.get_approx_embedding(soft_prob)

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

    def define_generator_as_hiddenLSTM(self, z=None, x=None, pretrain=False):
        """define the generator.

        Feed z as an init state to the cell,
        and feed the embedding of '<go>' as the input of the first lstm cell.
        """
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            tmp = tf.ones(
                (self.para.BATCH_SIZE, 1),
                dtype=tf.int32) * self.loader.vocab['<go>']
            input = self.embedding(tmp, reuse=True)[:, 0, :]
            state = tf.contrib.rnn.LSTMStateTuple(c=z, h=z)

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()

                cell_output, state = self.G_cell(
                    self.lstm.standard_lstm_unit, input, state)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = self.get_approx_embedding(soft_prob)

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
    def define_pretraining_op(self):
        # get the training vars for both networks
        vars_all = tf.trainable_variables()
        vars_D = [var for var in vars_all if 'discriminator' in var.name]
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # define optimizer
        optimizer_pretrain_D = self.define_optimizer(self.para.LEARNING_RATE_D)
        optimizer_pretrain_G = self.define_optimizer(self.para.LEARNING_RATE_G)

        # get grads and vars of D and G.
        self.grads_and_vars_pretrain_D \
            = optimizer_pretrain_D.compute_gradients(
                self.loss_pretrain_D, var_list=vars_D)
        self.grads_and_vars_pretrain_G \
            = optimizer_pretrain_G.compute_gradients(
                self.loss_pretrain_G, var_list=vars_G)

        # get training operation of G and D.
        self.op_pretrain_D = optimizer_pretrain_D.apply_gradients(
            self.grads_and_vars_pretrain_D)
        self.op_pretrain_G = optimizer_pretrain_G.apply_gradients(
            self.grads_and_vars_pretrain_G)

    def pretrain_step(self, sess, batch_x, batch_y, batch_ymask, losses):
        """pretrain the model."""
        feed_dict_D_pos = {
            self.x: batch_x,
            self.x_label: np.ones((self.para.BATCH_SIZE, 1)),
            self.dropout_val: self.para.DROPOUT_RATE}

        feed_dict_D_neg = {
            self.x: batch_x,
            self.x_label: np.ones((self.para.BATCH_SIZE, 1)),
            self.dropout_val: self.para.DROPOUT_RATE}

        feed_dict_G = {
            self.x: batch_x,
            self.y: batch_y,
            self.ymask: batch_ymask,
            self.dropout_val: self.para.DROPOUT_RATE}

        # train D.
        _, loss_D_pos, summary_D_pos = sess.run(
            [self.op_pretrain_D, self.loss_pretrain_D,
             self.pretrain_summary_D_op],
            feed_dict=feed_dict_D_pos)

        _, loss_D_neg, summary_D_neg = sess.run(
            [self.op_pretrain_D, self.loss_pretrain_D,
             self.pretrain_summary_D_op],
            feed_dict=feed_dict_D_neg)

        # train G.
        _, loss_G, summary_G = sess.run(
            [self.op_pretrain_G, self.loss_pretrain_G,
             self.pretrain_summary_G_op],
            feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D_pos + loss_D_neg)
        losses['losses_G'].append(loss_G)

        # summary
        self.pretrain_summary_writer.add_summary(summary_D_pos)
        self.pretrain_summary_writer.add_summary(summary_D_neg)
        self.pretrain_summary_writer.add_summary(summary_G)
        return losses

    def run_pretrain_epoch(self, sess):
        """run pretrain epoch."""
        losses = {'losses_D': [], 'losses_G': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()

        for step in range(self.loader.num_batches):
            batch_x, _, batch_y, batch_ymask = self.loader.next_batch()

            losses = self.pretrain_step(
                sess, batch_x, batch_y, batch_ymask, losses)

            sys.stdout.write(
                '\r{}/{}: pretrain: mean loss D = {}; mean loss G = {}'.format(
                    step, self.loader.num_batches,
                    np.mean(losses['losses_D']), np.mean(losses['losses_G']))
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = (end_epoch_time - start_epoch_time).seconds
        return np.mean(losses['losses_D']), \
            np.mean(losses['losses_G']), duration

    # define train related stuff
    def define_training_op(self):
        # get the training vars for both networks
        vars_all = tf.trainable_variables()
        vars_D = [var for var in vars_all if 'discriminator' in var.name]
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # define optimizer
        optimizer_D = self.define_optimizer(self.para.LEARNING_RATE_D)
        optimizer_G = self.define_optimizer(self.para.LEARNING_RATE_G)

        # get grads and vars of D and G.
        self.grads_and_vars_D \
            = optimizer_D.compute_gradients(self.loss_D, var_list=vars_D)
        self.grads_and_vars_G \
            = optimizer_G.compute_gradients(self.loss_G, var_list=vars_G)

        # get training operation of G and D.
        self.op_train_D = optimizer_D.apply_gradients(self.grads_and_vars_D)
        self.op_train_G = optimizer_G.apply_gradients(self.grads_and_vars_G)

    def train_step(self, sess, batch_x, batch_z, losses):
        """do the training step."""
        feed_dict_D = {
            self.x: batch_x, self.z: batch_z,
            self.dropout_val: self.para.DROPOUT_RATE}
        feed_dict_G = {
            self.z: batch_z,
            self.dropout_val: self.para.DROPOUT_RATE}

        # train D.
        for _ in range(self.para.D_ITERS_PER_BATCH):
            _, summary_D, loss_real_D, loss_fake_D, loss_D,\
                    predict_D_real, predict_D_fake = sess.run(
                        [self.op_train_D, self.train_summary_D_op,
                         self.loss_real_D, self.loss_fake_D, self.loss_D,
                         self.D_real, self.D_fake],
                        feed_dict=feed_dict_D)

            # record loss information of D.
            D_real_number = len(predict_D_real[predict_D_real > 0.5])
            D_fake_number = len(predict_D_fake[predict_D_fake < 0.5])

            losses['losses_real_D'].append(loss_real_D)
            losses['losses_fake_D'].append(loss_fake_D)
            losses['accuracy_D_real'].append(
                1.0 * D_real_number / self.para.BATCH_SIZE)
            losses['accuracy_D_fake'].append(
                1.0 * D_fake_number / self.para.BATCH_SIZE)
            losses['accuracy_D'].append(
                (D_real_number + D_fake_number) / self.para.BATCH_SIZE / 2.0)

        # train G.
        for _ in range(self.para.G_ITERS_PER_BATCH):
            _, summary_G, loss_G = sess.run(
                [self.op_train_G, self.train_summary_G_op, self.loss_G],
                feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)

        # summary
        self.summary_writer.add_summary(summary_D)
        self.summary_writer.add_summary(summary_G)
        return losses

    def run_train_epoch(self, sess, train=False, verbose=True):
        """run standard train epoch."""
        # define some basic parameters.
        losses = {
            'losses_real_D': [], 'losses_fake_D': [],
            'losses_D': [], 'losses_G': [],
            'accuracy_D_real': [], 'accuracy_D_fake': [], 'accuracy_D': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()
        for step in range(self.loader.num_batches):
            batch_x, batch_z, _, _ = self.loader.next_batch()
            losses = self.train_step(sess, batch_x, batch_z, losses)

            sys.stdout.write(
                "\r{}/{}: mean loss of D:{},\
                 mean accuracy of D:{},mean loss of G:{}".format(
                    step + 1,
                    self.loader.num_batches,
                    np.mean(losses['losses_D']),
                    np.mean(losses['accuracy_D']),
                    np.mean(losses['losses_G'])
                )
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = (end_epoch_time - start_epoch_time).seconds
        return np.mean(losses['losses_D']), \
            np.mean(losses['losses_G']), duration

    """define status tracking stuff."""
    # tracking status.
    def define_keep_tracking(self, sess):
        self.build_dir_for_tracking()
        self.keep_tracking_pretrain(sess)
        self.keep_tracking_train(sess)

    def build_dir_for_tracking(self):
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        self.out_dir = join(
            join(self.para.TRAINING_DIRECTORY,
                 "runs", auxi.get_fullname(self)),
            timestamp)

        # Checkpoint directory. Tensorflow assumes this directory
        # already exists so we need to create it
        checkpoint_dir = join(self.out_dir, "checkpoints")
        self.checkpoint_prefix = join(checkpoint_dir, "model")
        self.checkpoint_comparison = join(checkpoint_dir, "comparison")
        self.best_model = join(checkpoint_dir, "bestmodel")
        if not exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            os.makedirs(self.checkpoint_comparison)
        self.saver = tf.train.Saver(tf.global_variables())

    def keep_tracking_pretrain(self, sess):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_D = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_pretrain_D)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_pretrain_G)

        # Summaries for loss and accuracy
        loss_summary_D = tf.summary.scalar(
            "loss_pretrain_D", self.loss_pretrain_D)
        loss_summary_G = tf.summary.scalar(
            "loss_pretrain_G", self.loss_pretrain_G)

        # Train Summaries
        self.pretrain_summary_D_op = tf.summary.merge(
            [loss_summary_D, grad_summaries_merged_D])
        self.pretrain_summary_G_op = tf.summary.merge(
            [loss_summary_G, grad_summaries_merged_G])

        log("writing to {}\n".format(self.out_dir))
        train_summary_dir = join(self.out_dir, "summaries", "pretrain")
        self.pretrain_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

    # checking and recording part.
    def keep_tracking_train(self, sess):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_D = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_D)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_G)

        # Summaries for loss and accuracy
        loss_summary_D = tf.summary.scalar('loss_train_D', self.loss_D)
        loss_summary_G = tf.summary.scalar("loss_train_G", self.loss_G)

        # Train Summaries
        self.train_summary_D_op = tf.summary.merge(
            [loss_summary_D, grad_summaries_merged_D])
        self.train_summary_G_op = tf.summary.merge(
            [loss_summary_G, grad_summaries_merged_G])

        train_summary_dir = join(self.out_dir, "summaries", "train")
        self.summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

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
                raise 'an invaild usage.'

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

    def weighted_pick(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return int(np.searchsorted(t, np.random.rand(1) * s))

    """define stuff related to sentence generation, i.e., samping."""
    def beam_search_pick(self, probs):
        samples, scores = BeamSearch(probs).beamsearch(
            None, self.loader.vocab['<go>'], None,
            k=3, maxsample=len(probs), use_unk=False)
        sampleweights = samples[np.argmax(scores)]
        t = np.cumsum(sampleweights)
        s = np.sum(sampleweights)
        return int(np.searchsorted(t, np.random.rand(1) * s))

    # add function to generate sentence.
    def sample_from_latent_space(
            self, sess, vocab_word2index, vocab_index2word,
            sampling_type=1, pick=2):
        """generate sentence from latent space.."""
        state = sess.run(self.G_cell.zero_state(1, tf.float32))
        z = self.para.Z_PRIOR(size=(1, self.para.Z_DIM))
        input = z

        generated = []

        for n in range(self.para.SENTENCE_LENGTH_TO_GENERATE):
            if n == 0:
                probs, state = sess.run(
                    [self.G_prob, self.G_state],
                    {self.z: input, self.dropout_val: 1.0,
                     self.G_cell_init_state: state})
            else:
                probs, state = sess.run(
                    [self.yhat_prob, self.yhat_state],
                    {self.x: input, self.dropout_val: 1.0,
                     self.G_cell_init_state: state})

            p = probs[0, 0]

            if pick == 1:
                if sampling_type == 0:
                    sample_index = np.argmax(p)
                else:
                    sample_index = self.weighted_pick(p)
            elif pick == 2:
                sample_index = self.beam_search_pick(probs[0])

            pred = vocab_index2word[sample_index]
            word = pred
            generated.append(word)

            if word == '<eos>':
                break
            else:
                input = [[sample_index]]
        return generated
