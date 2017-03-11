# -*- coding: utf-8 -*-
import tensorflow as tf

from code.model.basicModel import BasicModel


class TextGANV1(BasicModel):
    """The textGAN here simply follow the paper.
    It uses an RNN network as the generator, an CNN as the discriminator.
    """

    def __init__(self, para, loader):
        """init parameters."""
        super(TextGANV1, self).__init__(para, loader)

        # init the basic model..
        self.define_rnn_cell()
        self.define_placeholder()

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
            self.D_real_logit, self.D_real \
                = self.define_discriminator_as_LSTM(embedded_x)

            # get discriminator on fake data. the reuse=True, which
            # specifies we reuse the discriminator ops for new placeholder.
            self.D_fake_logit, self.D_fake \
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
                    logits=self.D_real_logit,
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
