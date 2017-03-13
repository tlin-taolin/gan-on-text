# -*- coding: utf-8 -*-
import tensorflow as tf

from code.model.basicModel import BasicModel


class TextGAN(BasicModel):
    """The textGAN here simply follow the paper.
    It uses an RNN network as the generator, an CNN as the discriminator.
    """

    def __init__(self, para, loader, training=True):
        """init parameters."""
        super(TextGAN, self).__init__(para, loader, training)

        # init the basic model..
        self.define_placeholder()

        self.G_cell = self.define_rnn_cell('lstm')
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
        """define the train loss.

        For `sigmoid_cross_entropy_with_logits`, where z is label, x is data.
        we have z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)).
        """
        with tf.variable_scope("loss"):
            # D = \argmin_D - (E_{x \sim P_r} [\log D(x)] + E_{x \sim P_g} [\log (1 - D(x))])
            self.loss_real_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_real,
                    labels=tf.ones_like(self.D_logit_real)))

            self.loss_fake_D = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_fake,
                    labels=tf.zeros_like(self.D_logit_fake)))
            self.loss_D = self.loss_real_D + self.loss_fake_D

            # G loss: minimizes the divergence of D_logit_fake to 1 (real)
            # G = \argmin_G - E_{x \sim P_g} [\log (D(x))]
            self.loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_fake,
                    labels=tf.ones_like(self.D_logit_fake)))
