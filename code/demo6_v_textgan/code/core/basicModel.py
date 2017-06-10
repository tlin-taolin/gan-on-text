# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import sys
import time
import datetime
from os.path import join, exists

import numpy as np
import tensorflow as tf
from six.moves import cPickle

from code.utils.logger import log
import code.utils.auxiliary as auxi
import code.utils.opfiles as opfile


class BasicModel(object):
    """a base model for any subsequent implementation."""
    def __init__(self, para, loader, sess, infer):
        """init."""
        # system define.
        self.para = para
        self.sess = sess
        self.infer = infer
        self.loader = loader

        if infer:
            self.para.BATCH_SIZE = 1

    """define placeholder."""
    # define the basic components of the inference procedure.
    def define_placeholder(self):
        """define the placeholders."""
        self.enc_inputs = tf.placeholder(
            tf.int32, (self.para.BATCH_SIZE, None), name="enc_inputs")
        self.dec_inputs = tf.placeholder(
            tf.int32, (self.para.BATCH_SIZE, None), name="dec_inputs")
        self.targets = tf.placeholder(
            tf.int32, (self.para.BATCH_SIZE, None), name="targets")
        self.target_labels = tf.placeholder(
            tf.int32, (self.para.BATCH_SIZE, None), name="targets_label")
        self.dropout = tf.placeholder(tf.float32, (), name='dropout_prob')
        self.sent_len = tf.placeholder(tf.int32, (), name='sent_len')
        self.soft_argmax = tf.placeholder(tf.float32, (), name='soft_argmax')
        self.noise = tf.placeholder(
            tf.float32, (self.para.BATCH_SIZE, self.para.NOISE_DIM),
            name='noise')

    def define_pointer(self):
        self.batch_pointer = tf.Variable(
            0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(
            self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(
            0, name="epoch_pointer", trainable=False)

    """training/optimizer related."""
    def define_optimizer(self, learning_rate):
        if self.para.OPTIMIZER_NAME == 'Adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.para.OPTIMIZER_NAME == 'RMSProp':
            return tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise 'not a vaild optimizer.'

    def define_train_op(self):
        # get the training vars for both networks
        log('define train operations...')
        vars_all = tf.trainable_variables()

        vars_D = [var for var in vars_all if 'discriminator' in var.name]
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # define optimizer
        optimizer_D = self.define_optimizer(self.para.LEARNING_RATE_D)
        optimizer_G = self.define_optimizer(self.para.LEARNING_RATE_G)

        # define pretrain op
        grads_G, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_G_pre, vars_G), self.para.GRAD_CLIP)
        self.grads_and_vars_G_pre = zip(grads_G, vars_G)
        self.op_pretrain_G = optimizer_G.apply_gradients(
            self.grads_and_vars_G_pre)

        grads_D, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_D_pre, vars_D), self.para.GRAD_CLIP)
        self.grads_and_vars_D_pre = zip(grads_D, vars_D)
        self.op_pretrain_D = optimizer_D.apply_gradients(
            self.grads_and_vars_D_pre)

        # define clip operation.
        wgan_clip_values = map(float, self.para.WGAN_CLIP_VALUES.split(','))
        self.op_clip_D_vars = [
            var.assign(tf.clip_by_value(
                var, wgan_clip_values[0], wgan_clip_values[1])
            ) for var in vars_D]

        # define train op
        grads_and_vars_D = optimizer_D.compute_gradients(self.loss_D, vars_D)
        grads_and_vars_G = optimizer_G.compute_gradients(self.loss_G, vars_G)

        log('variables and gradients in the discriminator:')
        for var in vars_D:
            print('     {}'.format(var.name))
        for grad, var in grads_and_vars_D:
            grad_name = grad.name if grad is not None else None
            print('     grad:{}, var:{}'.format(grad_name, var.name))

        log('variables and gradients in the generator:')
        for var in vars_G:
            print('     {}'.format(var.name))
        for grad, var in grads_and_vars_G:
            grad_name = grad.name if grad is not None else None
            print('     grad:{}, var:{}'.format(grad_name, var.name))

        self.grads_and_vars_D = [
            (tf.clip_by_norm(grad, self.para.GRAD_CLIP), var)
            for grad, var in grads_and_vars_D]
        self.op_train_D = optimizer_D.apply_gradients(self.grads_and_vars_D)

        self.grads_and_vars_G = [
            (tf.clip_by_norm(grad, self.para.GRAD_CLIP), var)
            for grad, var in grads_and_vars_G]
        self.op_train_G = optimizer_G.apply_gradients(self.grads_and_vars_G)

    """define status tracking stuff."""
    def define_keep_tracking(self):
        self.build_dir_for_tracking()
        self.keep_tracking()

    def build_dir_for_tracking(self):
        # Output directory for models and summaries
        if self.para.INIT_FROM is None:
            timestamp = str(int(time.time()))
            datatype = self.loader.__class__.__name__
            parent_path = join(self.para.TRAIN_DIR, "runs", datatype)
            method_folder = join(parent_path, auxi.get_fullname(self))
            self.out_dir = join(method_folder, timestamp)
        else:
            self.out_dir = self.para.INIT_FROM

        # Checkpoint directory. Tensorflow assumes this directory
        # already exists so we need to create it
        self.checkpoint_dir = join(self.out_dir, "checkpoints")
        self.checkpoint_comparison = join(self.checkpoint_dir, "comparison")
        self.best_model = join(self.checkpoint_dir, "bestmodel")

        if not exists(self.checkpoint_dir):
            opfile.build_dirs(self.checkpoint_comparison)
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.para.MAXNUM_MODEL_TO_KEEP)

        # save configuration to the path.
        with open(join(self.checkpoint_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(self.para, f)

        parameters = '\n'.join(
            [k + '\t' + str(v) for k, v in self.para._get_kwargs()])
        opfile.write_txt(parameters, join(self.checkpoint_dir, 'config'))

    def keep_tracking(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_D = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_D)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_G)

        # Summaries for loss and accuracy
        loss_D_summary = tf.summary.scalar('loss_D', self.loss_D)
        loss_G_summary = tf.summary.scalar('loss_G', self.loss_G)

        # Summaries
        self.op_train_summary = tf.summary.merge(
            [loss_D_summary, loss_G_summary,
             grad_summaries_merged_D, grad_summaries_merged_G])

        # log("writing to {}\n".format(self.out_dir))
        train_summary_dir = join(self.out_dir, "summaries", "train")

        self.train_summary_writer = tf.summary.FileWriter(
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

    def save_model(self, model_index=[0]):
        self.saver.save(self.sess, self.best_model,
                        global_step=model_index[-1])
        model_index += [model_index[-1] + 1]
        log("save {}-th bestmodel to path: {}.\n".format(model_index[-1],
                                                         self.best_model))
        return model_index

    """define some common operations that are involved in inference."""
    def conv2d(self, x, W, strides, padding='SAME', name="conv"):
        """do convolution.
            x: [batch, in_height, in_width, in_channels]
            W: [filter_height, filter_width, in_channels, out_channels]
        """
        return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)

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

    def weight_variable_s(self, shape, name='W'):
        return tf.get_variable(
            name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name="b"):
        """init bias variable."""
        return tf.get_variable(
            name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def leakyrelu_s(self, x, alpha=1/5.5):
        return tf.maximum(x, alpha * x)

    def leakyrelu(self, conv, b, alpha=0.01, name="leaky_relu"):
        """use lrelu as the activation function."""
        tmp = tf.nn.bias_add(conv, b)
        return tf.maximum(tmp * alpha, tmp)

    def tanh(self, conv, b, name="tanh"):
        """use tanh as the activation function."""
        return tf.tanh(tf.nn.bias_add(conv, b), name=name)

    def get_scope_variable(
            self, scope_name, name, shape=None,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True):
        with tf.variable_scope(scope_name) as scope:
            try:
                v = tf.get_variable(name, shape, initializer=initializer)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable(name)
        return v
