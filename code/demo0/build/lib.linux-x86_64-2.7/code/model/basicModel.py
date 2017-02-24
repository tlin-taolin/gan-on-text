# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import os
import sys
import time
from os.path import join, exists

import numpy as np
import tensorflow as tf

import code.utils.auxiliary as auxi
import code.utils.initData as init_data
from code.utils.myprint import myprint


class BasicModel(object):
    """a base model for any subsequent implementation."""

    def __init__(self, para):
        """init."""
        # system define.
        self.para = para
        self.train_dir = para.TRAINING_DIRECTORY

    def keep_tracking_grad_and_vals(self, grad_summaries, grads_and_vars):
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        return grad_summaries

    def keep_tracking(self, sess):
        """keep track the status."""
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        grad_summaries = self.keep_tracking_grad_and_vals(
            grad_summaries, self.grads_and_vars_G)
        grad_summaries = self.keep_tracking_grad_and_vals(
            grad_summaries, self.grads_and_vars_D)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        self.out_dir = join(
            join(self.train_dir, "runs", auxi.get_fullname(self)), timestamp)
        myprint("writing to {}\n".format(self.out_dir))

        # Summaries for loss and accuracy
        loss_summary_G = tf.summary.scalar("loss_G", self.G_loss)
        loss_summary_D = tf.summary.scalar("loss_D", self.D_loss)

        # Train Summaries
        self.train_summary_op = tf.summary.merge(
            [loss_summary_G, loss_summary_D, grad_summaries_merged])
        train_summary_dir = join(self.out_dir, "summaries", "train")
        self.train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # dev summaries
        self.dev_summary_op = tf.summary.merge(
            [loss_summary_G, loss_summary_D])
        dev_summary_dir = join(self.out_dir, "summaries", "dev")
        self.dev_summary_writer = tf.summary.FileWriter(
            dev_summary_dir, sess.graph)

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

    def leakyrelu(self, x, alpha=1/5.5):
        return tf.maximum(x, alpha * x)

    def define_placeholder(self, shape_z, shape_x):
        """define the placeholders."""
        self.z = tf.placeholder(tf.float32, shape_z, name="z")
        self.x_real = tf.placeholder(tf.float32, shape_x, name="x_real")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    def define_parameters_totrack(self):
        """define some parameters that will be used in the future."""
        self.l2_loss = tf.constant(0.0)

    def define_loss(self):
        """calculate the rmse."""
        with tf.name_scope("loss"):
            # D loss
            self.loss_real_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_real_logit,
                    labels=tf.ones_like(self.D_real_logit)))
            self.loss_fake_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_fake_logit,
                    labels=tf.zeros_like(self.D_fake_logit)))
            self.loss_D = self.loss_real_D + self.loss_fake_D

            # G loss: minimizes the divergence of D_fake_logit to 1 (real)
            self.loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_fake_logit,
                    labels=tf.ones_like(self.D_fake_logit)))

    def define_training(self):
        # get the training vars for both networks
        vars_all = tf.trainable_variables()
        vars_D = [var for var in vars_all if 'discriminator' in var.name]
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # set up both optimizers to apply gradient descent w/ Adam
        # define optimizer
        self.optimizer_D = tf.train.AdamOptimizer(self.para.LEARNING_RATE_D)
        self.optimizer_G = tf.train.AdamOptimizer(self.para.LEARNING_RATE_G)

        # get grads and vars of D and G.
        self.grads_and_vars_D = self.optimizer_D.compute_gradients(
            self.loss_D, var_list=vars_D)
        self.grads_and_vars_G = self.optimizer_G.compute_gradients(
            self.loss_G, var_list=vars_G)
        # get training operation of G and D.
        self.op_D = self.optimizer_D.apply_gradients(self.grads_and_vars_D)
        self.op_G = self.optimizer_G.apply_gradients(self.grads_and_vars_G)

    def get_batches(self, real_x, z, shuffle=True):
        """get batch data."""
        batches = init_data.batch_data(
            self.para, real_x, z, shuffle=shuffle)
        num_batch = self.para.NUM_SAMPLES // self.para.BATCH_SIZE
        return batches, num_batch

    def train_step(self, sess, batch_real_x, batch_z, losses):
        """evaluate the training step."""

        # define feeding placeholders
        feed_dict_D = {
            self.x_real: batch_real_x,
            self.z: batch_z, self.dropout_keep_prob: 1.0}
        feed_dict_G = {
            self.z: batch_z, self.dropout_keep_prob: 1.0}

        # define D update procedure.
        for _ in range(self.para.D_ITERS_PER_G_ITERS):
            _, loss_real_D, loss_fake_D, loss_D = sess.run(
                [self.op_D, self.loss_real_D, self.loss_fake_D, self.loss_D],
                feed_dict_D)

        # define G update procedure.
        _, loss_G = sess.run([self.op_G, self.loss_G], feed_dict=feed_dict_G)

        # record loss.
        losses['losses_real_D'].append(loss_real_D)
        losses['losses_fake_D'].append(loss_fake_D)
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)
        return losses

    def run_epoch(self, sess, real_x, z, train=False, verbose=True):
        batches, num_batch = self.get_batches(real_x, z)

        # define some basic parameters.
        losses = {
            'losses_real_D': [], 'losses_fake_D': [],
            'losses_D': [], 'losses_G': []}

        for step, batch in enumerate(batches):
            batch_real_x, batch_z = batch
            losses = self.train_step(sess, batch_real_x, batch_z, losses)

            if verbose and step % verbose == 0 and train:
                sys.stdout.write(
                    "\r{} / {}: mean loss of D: {}, mean loss of G: {}".format(
                        step + 1, num_batch,
                        np.mean(losses['losses_D']),
                        np.mean(losses['losses_G'])
                    )
                )
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()
        return losses
