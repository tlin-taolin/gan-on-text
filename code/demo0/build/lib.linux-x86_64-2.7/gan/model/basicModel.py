# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import os
import sys
import time
from os.path import join, exists

import numpy as np
import tensorflow as tf

import gan.utils.auxiliary as auxi
import gan.utils.initData as init_data
from gan.utils.myprint import myprint


class BasicModel(object):
    """a base model for any subsequent implementation."""

    def __init__(self, para):
        """init."""
        # system define.
        self.para = para
        self.train_dir = para.TRAINING_DIRECTORY

    def leakyreulu(self, x, alpha=1/5.5):
        return tf.maximum(x, alpha * x)

    def define_placeholder(self, shape_z, shape_x):
        """define the placeholders."""
        self.z = tf.placeholder(tf.float32, shape_x, name="z")
        self.x_real = tf.placeholder(tf.float32, shape_x, name="x_real")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    def define_parameters_totrack(self):
        """define some parameters that will be used in the future."""
        self.l2_loss = tf.constant(0.0)

    def loss(self):
        """calculate the rmse."""
        with tf.name_scope("loss"):
            # D loss
            self.D_real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D_real_logit, tf.ones_like(self.D_real_logit)))
            self.D_fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D_fake_logit, tf.zeros_like(self.D_fake_logit)))
            self.D_loss = self.D_real_loss + self.D_fake_loss

            # G loss: minimizes the divergence of D_fake_logit to 1 (real)
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    self.D_fake_logit, tf.ones_like(self.D_fake_logit)))

    def training(self):
        # get the training vars for both networks
        t_vars = tf.trainable_variables()
        D_vars = [var for var in t_vars if 'discriminator' in var.name]
        G_vars = [var for var in t_vars if 'generator' in var.name]

        # set up both optimizers to apply gradient descent w/ Adam
        self.D_op = tf.train.AdamOptimizer(
            self.para.G_LEARNING_RATE).minimize(self.D_loss, var_list=D_vars)
        self.G_op = tf.train.AdamOptimizer(
            self.para.D_LEARNING_RATE).minimize(self.G_loss, var_list=G_vars)

    def get_batches(self, real_x, z, shuffle=True):
        """get batch data."""
        batches = init_data.batch_data(
            self.para, real_x, z, shuffle=shuffle)
        num_batch = self.para.NUM_SAMPLES // self.para.BATCH_SIZE
        return batches, num_batch

    def train_step(self, sess, batch_real_x, batch_z, losses):
        """evaluate the training step."""

        # define feeding placeholders
        D_feed_dict = {
            self.x_real: batch_real_x,
            self.z: batch_z, self.dropout_keep_prob: 1.0}
        G_feed_dict = {
            self.z: batch_z, self.dropout_keep_prob: 1.0}

        # define D update procedure.
        for _ in range(self.para.D_ITERS_PER_G_ITERS):
            _, D_real_loss, D_fake_loss, D_loss = sess.run(
                [self.D_op, self.D_real_loss, self.D_fake_loss, self.D_loss],
                D_feed_dict)

        # define G update procedure.
        _, G_loss = sess.run([self.G_op, self.G_loss], feed_dict=G_feed_dict)

        # record loss.
        losses['D_real_losses'].append(D_real_loss)
        losses['D_fake_losses'].append(D_fake_loss)
        losses['D_real_losses'].append(D_loss)
        losses['G_losses'].append(G_loss)

    def run_epoch(self, sess, real_x, z, train=False, verbose=True):
        batches, num_batch = self.get_batches(real_x, z)

        # define some basic parameters.
        losses = {
            'D_real_losses': [], 'D_fake_losses': [],
            'D_losses': [], 'G_losses': []}

        for step, batch in enumerate(batches):
            batch_real_x, batch_z = batch
            self.train_step(sess, batch_real_x, batch_z, losses)

            if verbose and step % verbose == 0 and train:
                sys.stdout.write(
                    '\r{} / {} : mean D loss = {}, mean G loss = {}'.format(
                        step + 1, num_batch,
                        np.mean(losses['D_losses']),
                        np.mean(losses['G_losses'])
                    )
                )
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()
        return losses
