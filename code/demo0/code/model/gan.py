# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from code.model.basicModel import BasicModel


class GAN(BasicModel):
    """define a simple GAN architecture."""

    def __init__(self, para):
        """init parameters."""
        super(GAN, self).__init__(para)

        self.define_placeholder(
            shape_z=[None, para.Z_DIM], shape_x=[None, para.X_DIM])
        self.define_parameters_totrack()

    def define_generator(
            self, z, out_dim=2, num_hidden_neuron=256, num_layers=2):
        """inference procedure of generative model."""
        with tf.variable_scope('generator'):
            hidden = z
            for hidden_idx in range(num_layers):
                hidden = fully_connected(
                    hidden, num_hidden_neuron, activation_fn=self.leakyrelu,
                    weights_initializer=tf.orthogonal_initializer(gain=1.4))
            x = fully_connected(
                hidden, out_dim, activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(gain=1.4))
        return x

    def define_discriminator(
            self, x, num_hidden_neuron=256, num_layers=2, reuse=False):
        """inference procedure of adversarial model."""
        # classifies whether x is real (1) or fake (0)
        # with a logistic regression output
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            hidden = x
            for h_idx in range(num_layers):
                hidden = fully_connected(
                    hidden, num_hidden_neuron, activation_fn=self.leakyrelu,
                    weights_initializer=tf.orthogonal_initializer(gain=1.4))
            logit = fully_connected(
                hidden, 1, activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(gain=1.4))
        return logit, tf.nn.sigmoid(logit)

    def define_inference(self):
        """define inference model."""
        # get generator ops
        self.G = self.define_generator(self.z)

        # get discriminator ops on real data.
        self.D_real_logit, D_real = self.define_discriminator(self.x_real)

        # get discriminator on fake data. The reuse=True, which
        # specifies we reuse the discriminator ops for this new placeholder.
        self.D_fake_logit, D_fake = self.define_discriminator(
            self.G, reuse=True)
