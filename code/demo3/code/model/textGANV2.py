# -*- coding: utf-8 -*-
import sys
import datetime
import numpy as np
import tensorflow as tf

from code.model.basicModel import BasicModel


class TextGANV2(BasicModel):
    """The textGAN here simply follow the paper.
    It uses an RNN network as the generator, an CNN as the discriminator.
    """

    def __init__(self, para, loader):
        """init parameters."""
        super(TextGANV2, self).__init__(para, loader)

        # init the basic model..
        self.define_rnn_cell()
        self.define_placeholder()

    def define_inference(self):
        """"define the inference procedure in training phase."""
        self.G_cell = self.define_rnn_cell()
        self.D_cell = self.define_rnn_cell()
        self.G_cell_init_state = self.G_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)
        self.D_cell_init_state = self.D_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)

        with tf.variable_scope('generator'):
            self.embedding()
            self.language_model()

            self.yhat_logit, self.yhat_prob, self.yhat_out, _ \
                = self.define_generator_as_LSTMV1(x=self.x, pretrain=True)
            self.G_logit, self.G_prob, self.G_out, self.G_embedded_out \
                = self.define_generator_as_LSTMV1(z=self.z, pretrain=False)
            embedded_x = self.embedding(self.x, reuse=True)

        with tf.variable_scope('discriminator'):
            self.D_logit_real, self.D_real \
                = self.define_discriminator_as_LSTM(embedded_x)

            # get discriminator on fake data. the reuse=True, which
            # specifies we reuse the discriminator ops for new placeholder.
            self.D_logit_fake, self.D_fake \
                = self.define_discriminator_as_LSTM(
                    self.G_embedded_out, reuse=True)

    def define_pretrain_loss(self):
        """define the pretrain loss.

        For `sigmoid_cross_entropy_with_logits`, where z is label, x is data.
        we have z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)).
        """
        with tf.name_scope("pretrain_loss"):
            # deal with discriminator.
            self.loss_pretrain_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_real,
                    labels=self.x_label
                )
            )

            self.loss_pretrain_G = tf.contrib.seq2seq.sequence_loss(
                logits=self.yhat_logit,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)

    def define_train_loss(self):
        """define the train loss."""
        with tf.variable_scope("loss"):
            self.loss_D = tf.reduce_mean(self.D_logit_real-self.D_logit_fake)
            self.loss_G = tf.reduce_mean(self.D_logit_fake)

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
            _, summary_D, loss_D, predict_D_real, predict_D_fake = sess.run(
                [self.op_D, self.train_summary_D_op, self.loss_D,
                 self.D_real, self.D_fake],
                feed_dict=feed_dict_D)

        # train G.
        for _ in range(self.para.G_ITERS_PER_BATCH):
            _, summary_G, loss_G = sess.run(
                [self.op_G, self.train_summary_G_op, self.loss_G],
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
            'losses_D': [], 'losses_G': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()
        for step in range(self.loader.num_batches):
            batch_x, batch_z, _, _ = self.loader.next_batch()
            losses = self.train_step(sess, batch_x, batch_z, losses)

            sys.stdout.write(
                "\r{}/{}: mean loss of D:{},mean loss of G:{}".format(
                    step + 1,
                    self.loader.num_batches,
                    np.mean(losses['losses_D']),
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