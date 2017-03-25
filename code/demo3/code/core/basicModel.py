# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import sys
import time
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
        grads_D, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_pretrain_D, vars_D), self.para.GRAD_CLIP)
        self.grads_and_vars_pretrain_D = zip(grads_D, vars_D)
        self.op_pretrain_D = optimizer_pretrain_D.apply_gradients(
            self.grads_and_vars_pretrain_D)

        grads_G, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_pretrain_G, vars_G), self.para.GRAD_CLIP)
        self.grads_and_vars_pretrain_G = zip(grads_G, vars_G)
        self.op_pretrain_G = optimizer_pretrain_G.apply_gradients(
            self.grads_and_vars_pretrain_G)

    def pretrain_step(self, losses):
        """pretrain the model."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()

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
        _, loss_D_pos, summary_D_pos = self.sess.run(
            [self.op_pretrain_D, self.loss_pretrain_D,
             self.pretrain_summary_D_op],
            feed_dict=feed_dict_D_pos)

        _, loss_D_neg, summary_D_neg = self.sess.run(
            [self.op_pretrain_D, self.loss_pretrain_D,
             self.pretrain_summary_D_op],
            feed_dict=feed_dict_D_neg)

        # train G.
        _, loss_G, summary_G = self.sess.run(
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

    def run_pretrain_epoch(self):
        """run pretrain epoch."""
        losses = {'losses_D': [], 'losses_G': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()

        for step in range(self.loader.num_batches):
            losses = self.pretrain_step(losses)

            sys.stdout.write(
                '\r{}/{}: pretrain: mean loss D = {}; mean loss G = {}'.format(
                    step, self.loader.num_batches,
                    np.mean(losses['losses_D']), np.mean(losses['losses_G']))
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = 1.0 * (
            end_epoch_time - start_epoch_time).seconds/self.loader.num_batches
        return np.mean(losses['losses_D']), \
            np.mean(losses['losses_G']), duration

    # define train related stuff
    def define_training_op(self):
        # get the training vars for both networks
        vars_all = tf.trainable_variables()
        vars_D = [var for var in vars_all if 'discriminator' in var.name]
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # define clip operation.
        self.op_clip_D_vars = [
            var.assign(tf.clip_by_value(
                var, self.para.WGAN_CLIP_VALUES[0],
                self.para.WGAN_CLIP_VALUES[1])) for var in vars_D]

        # define optimizer
        optimizer_D = self.define_optimizer(self.para.LEARNING_RATE_D)
        optimizer_G = self.define_optimizer(self.para.LEARNING_RATE_G)

        # get grads and vars of D and G.
        grads_D, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_pretrain_D, vars_D), self.para.GRAD_CLIP)
        self.grads_and_vars_D = zip(grads_D, vars_D)
        self.op_D = optimizer_D.apply_gradients(
            self.grads_and_vars_pretrain_D)

        grads_G, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_G, vars_G), self.para.GRAD_CLIP)
        self.grads_and_vars_G = zip(grads_G, vars_G)
        self.op_G = optimizer_G.apply_gradients(
            self.grads_and_vars_G)

    def train_step(self, losses):
        """do the training step."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()

        feed_dict_D = {
            self.x: batch_x, self.z: batch_z,
            self.dropout_val: self.para.DROPOUT_RATE}
        feed_dict_G = {
            self.z: batch_z,
            self.y: batch_y,
            self.ymask: batch_ymask,
            self.dropout_val: self.para.DROPOUT_RATE}

        # train D.
        for _ in range(self.para.D_ITERS_PER_BATCH):
            _, summary_D, loss_D_real, loss_D_fake, loss_D,\
                    predict_D_real, predict_D_fake = self.sess.run(
                        [self.op_train_D, self.train_summary_D_op,
                         self.loss_D_real, self.loss_D_fake, self.loss_D,
                         self.D_real, self.D_fake],
                        feed_dict=feed_dict_D)

            # record loss information of D.
            D_real_number = len(predict_D_real[predict_D_real > 0.5])
            D_fake_number = len(predict_D_fake[predict_D_fake < 0.5])

            losses['losses_D_real'].append(loss_D_real)
            losses['losses_D_fake'].append(loss_D_fake)
            losses['accuracy_D_real'].append(
                1.0 * D_real_number / self.para.BATCH_SIZE)
            losses['accuracy_D_fake'].append(
                1.0 * D_fake_number / self.para.BATCH_SIZE)
            losses['accuracy_D'].append(
                (D_real_number + D_fake_number) / self.para.BATCH_SIZE / 2.0)

        # train G.
        for _ in range(self.para.G_ITERS_PER_BATCH):
            _, summary_G, loss_G, perplexity_G = self.sess.run(
                [self.op_train_G, self.train_summary_G_op,
                 self.loss_G, self.perplexity_G],
                feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)
        losses['perplexity_G'].append(perplexity_G)

        # summary
        self.summary_writer.add_summary(summary_D)
        self.summary_writer.add_summary(summary_G)
        return losses

    def run_train_epoch(self):
        """run standard train epoch."""
        # define some basic parameters.
        losses = {
            'losses_D_real': [], 'losses_D_fake': [],
            'losses_D': [], 'losses_G': [], 'perplexity_G': [],
            'accuracy_D_real': [], 'accuracy_D_fake': [], 'accuracy_D': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()
        for step in range(self.loader.num_batches):
            losses = self.train_step(losses)

            sys.stdout.write(
                "\r{}/{}: mean loss of D: {}, mean accuracy of D: {}, mean loss of G: {}, mean perplexity of G: {}".format(
                    step + 1,
                    self.loader.num_batches,
                    np.mean(losses['losses_D']),
                    np.mean(losses['accuracy_D']),
                    np.mean(losses['losses_G']),
                    np.mean(losses['perplexity_G'])
                )
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = 1.0 * (
            end_epoch_time - start_epoch_time).seconds/self.loader.num_batches
        return np.mean(losses['losses_D']), np.mean(losses['losses_G']), duration

    """define status tracking stuff."""
    # tracking status.
    def define_keep_tracking(self):
        self.build_dir_for_tracking()
        self.keep_tracking_pretrain()
        self.keep_tracking_train()

    def build_dir_for_tracking(self):
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        datatype = self.loader.__class__.__name__
        parent_path = join(self.para.TRAINING_DIRECTORY, "runs", datatype)
        method_folder = join(parent_path, auxi.get_fullname(self))
        self.pretrain_dir = join(method_folder, 'pretrain')
        self.out_dir = join(method_folder, timestamp)

        # Checkpoint directory. Tensorflow assumes this directory
        # already exists so we need to create it
        checkpoint_dir = join(self.out_dir, "checkpoints")
        self.checkpoint_prefix = join(checkpoint_dir, "model")
        self.checkpoint_comparison = join(checkpoint_dir, "comparison")
        self.best_model = join(checkpoint_dir, "bestmodel")
        self.pretrain_model = join(self.pretrain_dir, "pretrainmodel")
        if not exists(checkpoint_dir):
            opfile.build_dirs(self.checkpoint_comparison)
            opfile.build_dirs(self.pretrain_dir)
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.para.MAX_MODEL_TO_KEEP)

    def keep_tracking_pretrain(self):
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
            train_summary_dir, self.sess.graph)

    # checking and recording part.
    def keep_tracking_train(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_D = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_D)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_G)

        # Summaries for loss and accuracy
        loss_summary_D = tf.summary.scalar('loss_train_D', self.loss_D)
        loss_summary_G = tf.summary.scalar("loss_train_G", self.loss_G)
        perplexity_summary_G = tf.summary.scalar(
            "perplexity_train_G", self.perplexity_G)

        # Train Summaries
        self.train_summary_D_op = tf.summary.merge(
            [loss_summary_D, grad_summaries_merged_D])
        self.train_summary_G_op = tf.summary.merge(
            [loss_summary_G, perplexity_summary_G, grad_summaries_merged_G])

        train_summary_dir = join(self.out_dir, "summaries", "train")
        self.summary_writer = tf.summary.FileWriter(
            train_summary_dir, self.sess.graph)

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
                soft_prob = tf.nn.softmax(logit * self.para.SOFT_ARGMAX)
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

    # add function to generate sentence.
    def sample_from_latent_space(self, vocab_word2index, vocab_index2word):
        """generate sentence from latent space.."""
        # init.
        state = self.sess.run(self.G_cell.zero_state(1, tf.float32))
        z = self.para.Z_PRIOR(size=(1, self.para.Z_DIM))
        input = z

        # some preparation
        # placeholders.
        generated_sentence = []
        word_search = WordSearch(self.para, self.loader.vocab)

        # generate first word
        log('sample from the continuous space.')
        log('...generate the first word.')
        probs, state, values, indices = self.sess.run(
            [self.G_prob, self.G_state, self.G_top_value, self.G_top_index],
            {self.z: input, self.dropout_val: 1.0,
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
                    [self.pre_prob, self.pre_state],
                    {self.x: input, self.dropout_val: 1.0,
                     self.G_cell_init_state: state})

                sample_index = basic_sampler(probs[0, 0])
                word = vocab_index2word[sample_index]
                generated_sentence.append(word)

                if word == '<eos>':
                    break
                else:
                    input = [[sample_index]]
            else:
                word_search.beam_search(
                    self.sess, self.x, self.dropout_val,
                    self.G_cell_init_state,
                    self.pre_top_value, self.pre_top_index, state)

            if len(word_search.beam_candidates) == 0:
                generated_sentence = [
                    vocab_index2word[s] for s in word_search.best_sequence]
                break
            if len(generated_sentence) > self.para.SENTENCE_LENGTH_TO_GENERATE:
                break
        return generated_sentence
