# -*- coding: utf-8 -*-
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

        self.G_cell = self.lstm.inherit_lstm_fn_from_tf(para, 'lstm')
        self.G_cell_init_state = self.G_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)

    def inference(self):
        """"define the inference procedure in training phase."""
        self.define_inference(
            self.define_generator_as_LSTM,
            self.define_discriminator_as_CNN)

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
        """define the train loss.

        For `sigmoid_cross_entropy_with_logits`, where z is label, x is data.
        we have z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)).
        """
        with tf.variable_scope("loss"):
            # D = \argmin_D - (E_{x \sim P_r} [\log D(x)] + E_{x \sim P_g} [\log (1 - D(x))])
            self.loss_D_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logit_D_real,
                    labels=tf.ones_like(self.logit_D_real)))

            self.loss_D_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logit_D_fake,
                    labels=tf.zeros_like(self.logit_D_fake)))
            self.loss_D = self.loss_D_real + self.loss_D_fake

            # G loss: minimizes the divergence of logit_D_fake to 1 (real)
            # G = \argmin_G - E_{x \sim P_g} [\log (D(x))]
            self.loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logit_D_fake,
                    labels=tf.ones_like(self.logit_D_fake)))

            self.perplexity_G = tf.contrib.seq2seq.sequence_loss(
                logits=self.G_logit,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)
