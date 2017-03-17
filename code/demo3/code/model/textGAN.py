# -*- coding: utf-8 -*-
import sys
import datetime
import numpy as np
import tensorflow as tf

from code.core.inferenceModel import InferenceModel
from code.core.lstm import LSTM


class TextGAN(InferenceModel):
    """The textGAN here simply follow the paper.
    It uses an RNN network as the generator, an CNN as the discriminator.
    """

    def __init__(self, para, loader, sess, training=True):
        """init parameters."""
        super(TextGAN, self).__init__(para, loader, sess, training)

        # init the basic model..
        self.define_placeholder()
        self.lstm = LSTM(para.RNN_SIZE, para.BATCH_SIZE)

        self.G_cell = self.lstm.cell
        self.D_cell = self.lstm.inherit_lstm_fn_from_tf(para, 'lstm')
        self.G_cell_init_state = self.lstm.init_state()
        self.D_cell_init_state = self.D_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)

    def inference(self):
        """"define the inference procedure in training phase."""
        self.define_inference(
            self.define_generator_as_hiddenLSTM,
            self.define_discriminator_as_LSTM)

    def define_pretrain_loss(self):
        """define the pretrain loss.

        For `sigmoid_cross_entropy_with_logits`, where z is label, x is data.
        we have z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)).
        """
        with tf.name_scope("pretrain_loss"):
            # deal with discriminator.
            self.loss_pretrain_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logit_D_real,
                    labels=self.x_label
                )
            )

            self.loss_pretrain_G = tf.contrib.seq2seq.sequence_loss(
                logits=self.pre_logit,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)

    def define_train_loss(self):
        """define the train loss."""
        with tf.variable_scope("loss"):
            self.loss_D = tf.reduce_mean(self.logit_D_fake - self.logit_D_real)
            self.loss_G = tf.reduce_mean(- self.logit_D_fake)

            self.perplexity_G = tf.contrib.seq2seq.sequence_loss(
                logits=self.G_logit,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)

    def train_step(self, losses):
        """do the training step."""
        batch_x, batch_z, batch_y, batch_ymask = self.loader.next_batch()

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
            _, summary_D, loss_D, predict_D_real, predict_D_fake = self.sess.run(
                [self.op_train_D, self.train_summary_D_op, self.loss_D,
                 self.D_real, self.D_fake],
                feed_dict=feed_dict_D)
            self.sess.run(self.op_clip_D_vars)

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
        losses = {'losses_D': [], 'losses_G': [], 'perplexity_G': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()
        for step in range(self.loader.num_batches):
            losses = self.train_step(losses)

            sys.stdout.write(
                "\r{}/{}: mean loss of D:{}, mean loss of G:{}, mean perplexity of G: {}".format(
                    step + 1,
                    self.loader.num_batches,
                    np.mean(losses['losses_D']),
                    np.mean(losses['losses_G']),
                    np.mean(losses['perplexity_G'])
                )
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = (end_epoch_time - start_epoch_time).seconds
        return np.mean(losses['losses_D']), \
            np.mean(losses['losses_G']), duration
