# -*- coding: utf-8 -*-
import sys
import datetime

import numpy as np
import tensorflow as tf

from code.core.inferenceModel import InferenceModel


class TextGANV0(InferenceModel):
    """The textG here simply follow the paper.
    It uses an RNN network as the generator.
    """

    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(TextGANV0, self).__init__(para, loader, sess, infer)

    def define_loss(self):
        """define the loss."""
        with tf.name_scope("loss"):
            self.loss_D = tf.reduce_mean(
                self.logits_D_real - self.logits_D_fake)
            self.loss_G = tf.reduce_mean(self.logits_D_fake)

            self.perplexity_G = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits_G,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)

    def train_step(self, global_step, losses):
        """train the model."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()
        soft_argmax = self.adjust_soft_argmax(global_step)

        feed_dict_D = {
            self.x_enc: [[t for t in l] for l, r in batch_x],
            self.x_dec: [[t for t in r] for l, r in batch_x],
            self.z: batch_z, self.y: batch_y,
            self.soft_argmax: soft_argmax
        }
        feed_dict_G = {
            self.x_enc: [[t for t in l] for l, r in batch_x],
            self.x_dec: [[t for t in r] for l, r in batch_x],
            self.z: batch_z, self.y: batch_y,
            self.soft_argmax: soft_argmax
        }

        # train D
        for _ in range(self.para.D_ITERS_PER_BATCH):
            _, summary_D, loss_D = self.sess.run(
                [self.op_train_D, self.op_train_summary_D, self.loss_D],
                feed_dict=feed_dict_D)
            self.sess.run(self.op_clip_D_vars)

        # train G.
        for _ in range(self.para.G_ITERS_PER_BATCH):
            _, summary_G, loss_G = self.sess.run(
                [self.op_train_G, self.op_train_summary_G, self.loss_G],
                feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)

        # summary
        self.train_summary_writer.add_summary(summary_D, global_step)
        self.train_summary_writer.add_summary(summary_G, global_step)
        return losses

    def val_step(self, global_step, losses):
        """validate the model."""
        batch_x, batch_y, batch_ymask, batch_z = self.loader.next_batch()
        soft_argmax = self.adjust_soft_argmax(global_step)

        feed_dict_D = {
            self.x_enc: [[t for t in l] for l, r in batch_x],
            self.x_dec: [[t for t in r] for l, r in batch_x],
            self.z: batch_z, self.y: batch_y,
            self.soft_argmax: soft_argmax
        }
        feed_dict_G = {
            self.x_enc: [[t for t in l] for l, r in batch_x],
            self.x_dec: [[t for t in r] for l, r in batch_x],
            self.z: batch_z, self.y: batch_y,
            self.soft_argmax: soft_argmax
        }

        # train D
        summary_D, loss_D = self.sess.run(
            [self.op_val_summary_D, self.loss_D], feed_dict=feed_dict_D)

        # train G.
        summary_G, loss_G, perplexity_G = self.sess.run(
            [self.op_val_summary_G, self.loss_G], feed_dict=feed_dict_G)

        # record loss.
        losses['losses_D'].append(loss_D)
        losses['losses_G'].append(loss_G)

        # summary
        self.val_summary_writer.add_summary(summary_D, global_step)
        self.val_summary_writer.add_summary(summary_G, global_step)
        return losses

    def run_epoch(self, stage, c_epoch, show_log=False, verbose=True):
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
            losses = stage(gloabl_step, losses)

            if show_log:
                sys.stdout.write(
                    "\r{}/{}: mean loss of D: {}, mean loss of G: {}".format(
                        step + 1,
                        self.loader.num_batches,
                        np.mean(losses['losses_D']),
                        np.mean(losses['losses_G'])
                    )
                )
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        end_epoch_time = datetime.datetime.now()
        duration = 1.0 * (
            end_epoch_time - start_epoch_time).seconds/self.loader.num_batches
        return np.mean(losses['losses_D']), np.mean(losses['losses_G']), duration
