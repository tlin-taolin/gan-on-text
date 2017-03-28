# -*- coding: utf-8 -*-
"""A basis class to inheritance."""
import sys
import time
import datetime
from os.path import join, exists

import numpy as np
import tensorflow as tf
from six.moves import cPickle

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
            self.para.SENTENCE_LENGTH = 1

    """define placeholder."""
    # define the basic components of the inference procedure.
    def define_placeholder(self):
        """define the placeholders."""
        self.x = tf.placeholder(
            tf.int32,
            [None, self.para.SENTENCE_LENGTH], name="x")
        self.x_label = tf.placeholder(
            tf.float32, [None, 1], name="x_label")
        self.y = tf.placeholder(
            tf.int32,
            [None, self.para.SENTENCE_LENGTH], name="y")
        self.ymask = tf.placeholder(
            tf.float32,
            [None, self.para.SENTENCE_LENGTH], name="ymask")
        self.z = tf.placeholder(
            tf.float32,
            [None, self.para.EMBEDDING_SIZE], name='z')
        self.soft_argmax = tf.placeholder(
            tf.float32, name='soft_argmax')

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
        vars_all = tf.trainable_variables()
        vars_D = [var for var in vars_all if 'discriminator' in var.name]
        vars_G = [var for var in vars_all if 'generator' in var.name]

        # define optimizer
        optimizer_D = self.define_optimizer(self.para.LEARNING_RATE_D)
        optimizer_G = self.define_optimizer(self.para.LEARNING_RATE_G)

        # define pretrain op
        grads_G, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_G_pretrain, vars_G), self.para.GRAD_CLIP)
        self.grads_and_vars_G_pretrain = zip(grads_G, vars_G)
        self.op_pretrain_G = optimizer_G.apply_gradients(
            self.grads_and_vars_G_pretrain)

        grads_D, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_D_pretrain, vars_D), self.para.GRAD_CLIP)
        self.grads_and_vars_D_pretrain = zip(grads_D, vars_D)
        self.op_pretrain_D = optimizer_D.apply_gradients(
            self.grads_and_vars_D_pretrain)

        # define clip operation.
        wgan_clip_values = map(float, self.para.WGAN_CLIP_VALUES.split(','))
        self.op_clip_D_vars = [
            var.assign(tf.clip_by_value(
                var, wgan_clip_values[0], wgan_clip_values[1])
            ) for var in vars_D]

        # define train op
        grads_G, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_G, vars_G), self.para.GRAD_CLIP)
        self.grads_and_vars_G = zip(grads_G, vars_G)
        self.op_train_G = optimizer_G.apply_gradients(self.grads_and_vars_G)

        grads_D, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss_D, vars_D), self.para.GRAD_CLIP)
        self.grads_and_vars_D = zip(grads_D, vars_D)
        self.op_train_D = optimizer_D.apply_gradients(self.grads_and_vars_D)

    def train_step(self, global_step, losses):
        """train the model."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()
        soft_argmax = self.adjust_soft_argmax(global_step)

        feed_dict_D = {
            self.x: batch_x, self.z: batch_z, self.soft_argmax: soft_argmax}
        feed_dict_G = {
            self.z: batch_z, self.y: batch_y, self.ymask: batch_ymask,
            self.soft_argmax: soft_argmax}

        # train D
        for _ in range(self.para.D_ITERS_PER_BATCH):
            _, summary_D, loss_D_real, loss_D_fake, loss_D,\
                predict_D_real, predict_D_fake = self.sess.run(
                    [self.op_train_D, self.op_train_summary_D,
                     self.loss_D_real, self.loss_D_fake, self.loss_D,
                     self.D_real, self.D_fake],
                    feed_dict=feed_dict_D)

            # record loss information of D.
            D_real_number = len(predict_D_real[predict_D_real > 0.5])
            D_fake_number = len(predict_D_fake[predict_D_fake < 0.5])

            losses['losses_D_real'].append(loss_D_real)
            losses['losses_D_fake'].append(loss_D_fake)
            losses['accuracy_D_real'].append(
                1.0 * D_real_number / self.para.BATCH_SIZE)
            losses['accuracy_D_fake'].append(
                1.0 * D_fake_number / self.para.BATCH_SIZE)
            losses['accuracy_D'].append(
                (D_real_number + D_fake_number) / self.para.BATCH_SIZE / 2.0)

        # train G.
        for _ in range(self.para.G_ITERS_PER_BATCH):
            _, summary_G, loss_G, perplexity_G = self.sess.run(
                [self.op_train_G, self.op_train_summary_G,
                 self.loss_G, self.perplexity_G],
                feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)
        losses['perplexity_G'].append(perplexity_G)

        # summary
        self.train_summary_writer.add_summary(summary_D, global_step)
        self.train_summary_writer.add_summary(summary_G, global_step)
        return losses

    def val_step(self, global_step, losses):
        """validate the model."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()
        soft_argmax = self.adjust_soft_argmax(global_step)

        feed_dict_D = {
            self.x: batch_x, self.z: batch_z, self.soft_argmax: soft_argmax}
        feed_dict_G = {
            self.z: batch_z, self.y: batch_y, self.ymask: batch_ymask,
            self.soft_argmax: soft_argmax}

        # val D
        summary_D, loss_D_real, loss_D_fake, loss_D,\
            predict_D_real, predict_D_fake = self.sess.run(
                [self.op_val_summary_D, self.loss_D_real, self.loss_D_fake,
                 self.loss_D, self.D_real, self.D_fake],
                feed_dict=feed_dict_D)

        # record loss information of D.
        D_real_number = len(predict_D_real[predict_D_real > 0.5])
        D_fake_number = len(predict_D_fake[predict_D_fake < 0.5])

        losses['losses_D_real'].append(loss_D_real)
        losses['losses_D_fake'].append(loss_D_fake)
        losses['accuracy_D_real'].append(
            1.0 * D_real_number / self.para.BATCH_SIZE)
        losses['accuracy_D_fake'].append(
            1.0 * D_fake_number / self.para.BATCH_SIZE)
        losses['accuracy_D'].append(
            (D_real_number + D_fake_number) / self.para.BATCH_SIZE / 2.0)

        # val G.
        summary_G, loss_G, perplexity_G = self.sess.run(
            [self.op_val_summary_G, self.loss_G, self.perplexity_G],
            feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)
        losses['perplexity_G'].append(perplexity_G)

        # summary
        self.val_summary_writer.add_summary(summary_D, global_step)
        self.val_summary_writer.add_summary(summary_G, global_step)
        return losses

    def run_epoch(self, stage, c_epoch, show_log=False, verbose=True):
        """run pretrain epoch."""
        losses = {
            'losses_D_real': [], 'losses_D_fake': [],
            'losses_D': [], 'losses_G': [], 'perplexity_G': [],
            'accuracy_D_real': [], 'accuracy_D_fake': [], 'accuracy_D': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()

        # determine start point
        if self.para.INIT_FROM is None and 'train' in stage.__name__:
            assign_op = self.batch_pointer.assign(0)
            self.sess.run(assign_op)
            assign_op = self.epoch_pointer.assign(c_epoch)
            self.sess.run(assign_op)
        if self.para.INIT_FROM is not None:
            self.loader.pointer = self.batch_pointer.eval()
            self.para.INIT_FROM = None

        batch_scope = self.loader.determine_batch_pointer_pos(stage)

        for step in batch_scope:
            gloabl_step = c_epoch * self.loader.num_batches + step
            losses = stage(gloabl_step, losses)

            if show_log:
                sys.stdout.write(
                    "\r{}/{}: mean loss of D: {}, mean accuracy of D: {}, mean loss of G: {}, mean perplexity of G: {}".format(
                        step + 1,
                        self.loader.num_batches,
                        np.mean(losses['losses_D']),
                        np.mean(losses['accuracy_D']),
                        np.mean(losses['losses_G']),
                        np.mean(losses['perplexity_G'])
                    )
                )
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = 1.0 * (
            end_epoch_time - start_epoch_time).seconds/self.loader.num_batches
        return np.mean(losses['losses_D']), \
            np.mean(losses['losses_G']), duration

    def pretrain_step(self, global_step, losses):
        """train the model."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()

        feed_dict_D_pos = {
            self.x: batch_x,
            self.x_label: np.ones((self.para.BATCH_SIZE, 1), dtype=np.float32)}
        feed_dict_D_neg = {
            self.x: self.loader.swap_x(batch_x),
            self.x_label: np.ones((self.para.BATCH_SIZE, 1), dtype=np.float32)}
        feed_dict_G = {
            self.x: batch_x, self.y: batch_y, self.ymask: batch_ymask}

        # train D
        _, loss_D_pos, summary_D_pos = self.sess.run(
            [self.op_pretrain_D, self.loss_D_pretrain,
             self.op_pretrain_summary_D],
            feed_dict=feed_dict_D_pos)

        _, loss_D_neg, summary_D_neg = self.sess.run(
            [self.op_pretrain_D, self.loss_D_pretrain,
             self.op_pretrain_summary_D],
            feed_dict=feed_dict_D_neg)

        # train G.
        _, loss_G, summary_G = self.sess.run(
            [self.op_pretrain_G, self.loss_G_pretrain,
             self.op_pretrain_summary_G],
            feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D_pos + loss_D_neg)
        losses['losses_G'].append(loss_G)

        # summary
        self.pretrain_summary_writer.add_summary(summary_D_pos)
        self.pretrain_summary_writer.add_summary(summary_D_neg)
        self.pretrain_summary_writer.add_summary(summary_G)
        return losses

    def run_epoch_pretrain(self, stage, c_epoch):
        """run pretrain epoch."""
        losses = {'losses_D': [], 'losses_G': []}

        start_epoch_time = datetime.datetime.now()
        self.loader.reset_batch_pointer()

        # determine start point
        if self.para.INIT_FROM is None and 'train' in stage.__name__:
            assign_op = self.batch_pointer.assign(0)
            self.sess.run(assign_op)
            assign_op = self.epoch_pointer.assign(c_epoch)
            self.sess.run(assign_op)
        if self.para.INIT_FROM is not None:
            self.loader.pointer = self.batch_pointer.eval()
            self.para.INIT_FROM = None

        batch_scope = self.loader.determine_batch_pointer_pos(stage)

        for step in batch_scope:
            gloabl_step = c_epoch * self.loader.num_batches + step
            losses = self.pretrain_step(gloabl_step, losses)

            sys.stdout.write(
                '\r{}/{}: pretrain: mean loss D = {}; mean loss G = {}'.format(
                    step, self.loader.num_batches,
                    np.mean(losses['losses_D']), np.mean(losses['losses_G']))
            )
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = 1.0 * (
            end_epoch_time - start_epoch_time).seconds/self.loader.num_batches
        return np.mean(losses['losses_D']), \
            np.mean(losses['losses_G']), duration

    def adjust_soft_argmax(self, global_step):
        """adjust soft_argmax parameter."""
        cur_epoch = global_step // self.loader.num_batches
        min_epoch = self.para.EPOCH_PRETRAIN
        duration_epoch = self.para.EPOCH_TRAIN
        soft_argmax_range = map(float, self.para.SOFT_ARGMAX.split(','))
        soft_argmax_lb, soft_argmax_ub = soft_argmax_range
        soft_argmax_step = (soft_argmax_ub - soft_argmax_lb) / duration_epoch
        cur_soft_argmax = \
            soft_argmax_lb + (cur_epoch - min_epoch) * soft_argmax_step
        cur_soft_argmax = cur_soft_argmax \
            if cur_soft_argmax >= soft_argmax_lb else soft_argmax_lb
        return cur_soft_argmax

    """define status tracking stuff."""
    def define_keep_tracking(self):
        self.build_dir_for_tracking()
        self.keep_tracking_pretrain()
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

    def keep_tracking_pretrain(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_D = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_D_pretrain)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_G_pretrain)

        # Summaries for loss and accuracy
        loss_summary_D = tf.summary.scalar(
            "loss_pretrain_D", self.loss_D_pretrain)
        loss_summary_G = tf.summary.scalar(
            "loss_pretrain_G", self.loss_G_pretrain)

        # Train Summaries
        self.op_pretrain_summary_D = tf.summary.merge(
            [loss_summary_D, grad_summaries_merged_D])
        self.op_pretrain_summary_G = tf.summary.merge(
            [loss_summary_G, grad_summaries_merged_G])

        pretrain_summary_dir = join(self.out_dir, "summaries", "pretrain")
        self.pretrain_summary_writer = tf.summary.FileWriter(
            pretrain_summary_dir, self.sess.graph)

    def keep_tracking(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged_D = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_D)
        grad_summaries_merged_G = self.keep_tracking_grad_and_vals(
            self.grads_and_vars_G)

        # Summaries for loss and accuracy
        loss_summary_D = tf.summary.scalar('loss_D', self.loss_D)
        loss_summary_G = tf.summary.scalar("loss_G", self.loss_G)
        perplexity_G = tf.summary.scalar("perplexity", self.perplexity_G)

        # Summaries
        self.op_train_summary_D = tf.summary.merge(
            [loss_summary_D, grad_summaries_merged_D])
        self.op_val_summary_D = tf.summary.merge([loss_summary_D])
        self.op_train_summary_G = tf.summary.merge(
            [loss_summary_G, perplexity_G, grad_summaries_merged_G])
        self.op_val_summary_G = tf.summary.merge([loss_summary_G])

        # log("writing to {}\n".format(self.out_dir))
        train_summary_dir = join(self.out_dir, "summaries", "train")
        val_summary_dir = join(self.out_dir, "summaries", "val")

        self.train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(
            val_summary_dir, self.sess.graph)

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

    def bias_variable(self, shape, name="b"):
        """init bias variable."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def leakyrelu_s(self, x, alpha=1/5.5):
        return tf.maximum(x, alpha * x)

    def leakyrelu(self, conv, b, alpha=0.01, name="leaky_relu"):
        """use lrelu as the activation function."""
        tmp = tf.nn.bias_add(conv, b)
        return tf.maximum(tmp * alpha, tmp)

    def tanh(self, conv, b, name="tanh"):
        """use tanh as the activation function."""
        return tf.tanh(tf.nn.bias_add(conv, b), name=name)
