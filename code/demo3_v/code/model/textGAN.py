# -*- coding: utf-8 -*-
import tensorflow as tf

from code.core.inferenceModel import InferenceModel


class TextGAN(InferenceModel):
    """The textG here simply follow the paper.
    It uses an RNN network as the generator.
    """

    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(TextGAN, self).__init__(para, loader, sess, infer)

    def define_loss(self):
        """define the loss."""
        with tf.name_scope("loss"):
            # D = \argmin_D - (E_{x \sim P_r} [\log D(x)] + E_{x \sim P_g} [\log (1 - D(x))])
            self.loss_D_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_D_real,
                    labels=tf.ones_like(self.logits_D_real)))

            self.loss_D_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_D_fake,
                    labels=tf.zeros_like(self.logits_D_fake)))

            self.loss_D = self.loss_D_real + self.loss_D_fake

            # G loss: minimizes the divergence of logits_D_fake to 1 (real)
            # G = \argmin_G - E_{x \sim P_g} [\log (D(x))]
            self.loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_D_fake,
                    labels=tf.ones_like(self.logits_D_fake)
                )
            )

            self.perplexity_G = tf.pow(
                tf.contrib.seq2seq.sequence_loss(
                    logits=self.logits_G,
                    targets=self.y,
                    weights=self.ymask,
                    average_across_timesteps=True,
                    average_across_batch=True),
                2)
