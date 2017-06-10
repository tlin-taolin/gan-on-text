# -*- coding: utf-8 -*-
import sys
import datetime

import numpy as np
import tensorflow as tf


from code.utils.logger import log
from code.core.inferenceModel import InferenceModel


class TextGANV3(InferenceModel):
    """The textG here simply follow the paper.
    It uses an RNN network as the generator.
    """

    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(TextGANV3, self).__init__(para, loader, sess, infer)
        self.para.PROJECTION_SIZE = self.para.EMBEDDING_SIZE

    """model definition part."""
    def prepare_generator(self):
        """prepare the generator."""
        self.gen_lstm = self.lstm(self.para)
        self.gen_cell = self.gen_lstm.inherit_lstm_fn_from_tf(
            self.para.RNN_TYPE, self.para.RNN_SIZE,
            self.para.PROJECTION_SIZE, self.dropout)
        self.gen_init_state = self.gen_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)

    def define_generator(self, inputs, teacher_forced):
        """define the generator."""
        if teacher_forced:
            outputs, state = tf.nn.dynamic_rnn(
                cell=self.gen_cell,
                inputs=inputs,
                initial_state=self.gen_init_state,
                dtype=tf.float32,
                time_major=False)
            return outputs, state, None
        else:
            # define the usage of while_loop.
            step = tf.constant(0)
            input = inputs[:, 0, :]
            # the outputs here represent the projected hidden states.
            next_inputs = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            log_probs = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            state = self.gen_init_state

            def should_continue(index, *args):
                return index < self.sent_len

            def body(step, input, log_probs, outputs, state):
                # inference procedure.
                with tf.variable_scope('rnn'):
                    tf.get_variable_scope().reuse_variables()
                    output, state = self.gen_cell(input, state)

                _, _, log_prob, soft_prob = self.language_model(output)

                next_input = self.get_approx_embedding(soft_prob)
                outputs = outputs.write(step, next_input)
                log_probs = log_probs.write(step, log_prob)
                return step + 1, next_input, log_probs, outputs, state

            _, _, log_probs, next_inputs, state = tf.while_loop(
                should_continue, body,
                [step, input, log_probs, next_inputs, state])

            next_inputs = tf.reshape(
                tf.transpose(next_inputs.stack(), perm=[1, 0, 2]),
                [self.para.BATCH_SIZE, -1, self.para.EMBEDDING_SIZE])
            return next_inputs, state, log_probs

    def prepare_discriminator(self):
        self.dis_lstm = self.lstm(self.para)
        self.dis_cell = self.dis_lstm.inherit_lstm_fn_from_tf(
            self.para.RNN_TYPE, self.para.RNN_SIZE,
            self.para.PROJECTION_SIZE, self.dropout)

    def define_discriminator(self, inputs, reuse=False):
        """define the discriminator."""
        if reuse:
            tf.get_variable_scope().reuse_variables()

        outputs, state = tf.nn.dynamic_rnn(
            cell=self.dis_cell,
            inputs=inputs,
            dtype=tf.float32,
            time_major=False)

        # output the classification result.
        output = outputs[:, -1, :]
        W = self.get_scope_variable(
            'output', 'W',
            shape=[self.para.EMBEDDING_SIZE, self.para.LABEL_DIM])
        b = self.get_scope_variable(
            'output', 'b',
            shape=[self.para.LABEL_DIM])

        logits = tf.nn.xw_plus_b(output, W, b, name="scores")
        return logits, tf.nn.sigmoid(logits)

    def define_summarize_function(self, inputs, outputs):
        """fuse the inputs and outputs."""
        fused = tf.concat([inputs, outputs], 2)
        return fused
        # w = self.get_scope_variable(
        #     'summary', 'w',
        #     shape=[self.para.EMBEDDING_SIZE * 2, self.para.EMBEDDING_SIZE])
        # b = self.get_scope_variable(
        #     'summary', 'b',
        #     shape=[self.para.EMBEDDING_SIZE])
        # fused = tf.reshape(fused, [-1, self.para.EMBEDDING_SIZE * 2])
        # logit = tf.matmul(fused, w) + b
        # logit = tf.reshape(
        #     logit, [self.para.BATCH_SIZE, -1, self.para.EMBEDDING_SIZE])
        # return tf.tanh(logit)

    def define_inference(self):
        """define inference procedure."""
        with tf.variable_scope('generator'):
            log('define generator.')

            # init some function.
            self.prepare_generator()

            inputs_emb = self.embedding_model(self.dec_inputs)

            # teacher-forcing.
            g_outputs_t, g_state_t, _ = self.define_generator(
                inputs_emb, teacher_forced=True)
            g_logits_t, _, _, _ = self.language_model(g_outputs_t)
            self.g_logits_t = g_logits_t

            # free-running.
            g_outputs_f, g_state_f, _ = self.define_generator(
                inputs_emb, teacher_forced=False)

        with tf.variable_scope('discriminator'):
            log('define discriminator.')

            # define the inputs.
            true_dis_inputs = self.define_summarize_function(
                inputs_emb, g_outputs_t)
            fake_dis_inputs = self.define_summarize_function(
                inputs_emb, g_outputs_f)

            # feed the inputs.
            self.prepare_discriminator()

            self.logits_D_real, self.D_real = \
                self.define_discriminator(true_dis_inputs)
            self.logits_D_fake, self.D_fake = \
                self.define_discriminator(fake_dis_inputs, reuse=True)

    def define_loss(self):
        """define the loss."""
        with tf.name_scope("pretrain_loss"):
            self.loss_D_pre = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_D_real,
                    labels=tf.cast(self.target_labels, tf.float32)
                )
            )

            self.loss_G_pre = tf.contrib.seq2seq.sequence_loss(
                logits=tf.reshape(
                    self.g_logits_t,
                    [self.para.BATCH_SIZE, self.sent_len,
                     self.loader.vocab_size]),
                targets=self.targets,
                weights=tf.cast(self.targets, tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)

        with tf.name_scope("train_loss"):
            loss_G = - tf.reduce_mean(self.logits_D_fake) + self.loss_G_pre
            loss_D = - tf.reduce_mean(self.logits_D_real - self.logits_D_fake)
            self.loss_G, self.loss_D = loss_G, loss_D

    """model running part."""
    def running(self):
        log('------ training ------ \n')
        upper_epoches = self.para.EPOCH_PRETRAIN + self.para.EPOCH_TRAIN

        for c_epoch in range(self.epoch_pointer.eval() + 1, upper_epoches):

            log('train epoch {}'.format(c_epoch))
            self.train_step(c_epoch)

            if c_epoch % self.para.CHECKPOINT_EVERY == 0:
                self.save_model()

        log('------ save the final model ------ \n')
        self.save_model()

    def train_step(self, c_epoch):
        start_epoch_time = datetime.datetime.now()
        losses = {'losses_D': [], 'losses_G': []}
        cur_batch = 0

        while True:
            data, noise, index = self.loader.next_batch()
            soft_argmax = self.adjust_soft_argmax(c_epoch)

            if data is None:
                break

            # load data.
            _, inp, tar = self.loader.process_bucket(data, index)

            feed_dict_D = {
                self.dec_inputs: inp,
                self.targets: tar,
                self.sent_len: inp.shape[1],
                self.dropout: self.para.DROPOUT_RATE,
                self.soft_argmax: soft_argmax
            }
            feed_dict_G = {
                self.dec_inputs: inp,
                self.targets: tar,
                self.sent_len: inp.shape[1],
                self.dropout: self.para.DROPOUT_RATE,
                self.soft_argmax: soft_argmax
            }

            # train D.
            for _ in range(self.para.D_ITERS_PER_BATCH):
                _, loss_D, _ = self.sess.run(
                    [self.op_train_D, self.loss_D, self.op_clip_D_vars],
                    feed_dict_D)
                losses['losses_D'].append(loss_D)

            # train G.
            for _ in range(self.para.G_ITERS_PER_BATCH):
                _, loss_G = self.sess.run(
                    [self.op_train_G, self.loss_G], feed_dict_G)
                losses['losses_G'].append(loss_G)

            sys.stdout.write(
                '\rcurrent batch:{}, loss D:{}, loss G:{}'.format(
                    cur_batch, np.mean(losses['losses_D']),
                    np.mean(losses['losses_G'])))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
        end_epoch_time = datetime.datetime.now()
        speed = 1.0 * (end_epoch_time - start_epoch_time).seconds / cur_batch

        log('loss D:{}, loss G:{}, speed: {:.2f} seconds/batch.'.format(
            np.mean(losses['losses_D']), np.mean(losses['losses_G']), speed))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
