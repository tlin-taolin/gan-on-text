# -*- coding: utf-8 -*-
import sys
import datetime

import numpy as np
import tensorflow as tf


from code.utils.logger import log
from code.core.inferenceModel import InferenceModel


class TextGANV1(InferenceModel):
    """The textG here simply follow the paper.
    It uses an RNN network as the generator.
    """

    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(TextGANV1, self).__init__(para, loader, sess, infer)

    """model definition part."""
    def prepare_generator(self):
        """prepare the generator."""
        self.gen_lstm = self.lstm(self.para)
        self.gen_cell = self.gen_lstm.inherit_lstm_fn_from_tf(
            self.para.RNN_TYPE, self.para.RNN_SIZE,
            self.para.PROJECTION_SIZE, self.dropout)
        self.gen_init_state = self.gen_cell.zero_state(
            self.para.BATCH_SIZE, tf.float32)

    def define_generator(self, inputs, pretrain=False):
        """define the generator."""
        if pretrain:
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
            outputs = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
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

                input = self.get_approx_embedding(soft_prob)
                outputs = outputs.write(step, input)
                log_probs = log_probs.write(step, log_prob)
                return step + 1, input, log_probs, outputs, state

            _, _, log_probs, outputs, state = tf.while_loop(
                should_continue, body,
                [step, input, log_probs, outputs, state])

            outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])
            outputs = tf.reshape(
                outputs, [self.para.BATCH_SIZE, -1, self.para.EMBEDDING_SIZE])
            return outputs, state, log_probs

    def define_discriminator(self, inputs, reuse=False):
        """define the discriminator."""
        if reuse:
            tf.get_variable_scope().reuse_variables()

        inputs = tf.expand_dims(inputs, 3)

        # build a series for 'CONV + RELU + POOL' architecture.
        D_CONV_SPATIALS = map(int, self.para.D_CONV_SPATIALS.split(','))
        D_CONV_DEPTHS = map(int, self.para.D_CONV_DEPTHS.split(','))
        archits = zip(D_CONV_SPATIALS, D_CONV_DEPTHS)

        pooled_outputs = []
        for i, (conv_spatial, conv_depth) in enumerate(archits):
            with tf.variable_scope("conv-lrelu-pooling-%s" % i):
                W = self.weight_variable(
                    shape=[
                        conv_spatial,
                        self.para.EMBEDDING_SIZE,
                        1,
                        conv_depth])
                b = self.bias_variable(shape=[conv_depth])

                conv = self.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                h = self.leakyrelu(conv, b, alpha=1.0/5.5)
                pooled = tf.reduce_max(h, axis=1)
                pooled_outputs.append(pooled)

        num_filters_total = sum([x[1] for x in archits])
        h_pool_flat = tf.reshape(
            tf.concat(pooled_outputs, 2), [-1, num_filters_total])

        # output the classification result.
        W = self.get_scope_variable(
            'output', 'W',
            shape=[num_filters_total, self.para.LABEL_DIM])
        b = self.get_scope_variable(
            'output', 'b',
            shape=[self.para.LABEL_DIM])

        logits = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
        return logits, tf.nn.sigmoid(logits)

    def define_inference(self):
        """define inference procedure."""
        with tf.variable_scope('generator'):
            log('define generator.')

            # init some function.
            self.prepare_generator()

            inputs_emb = self.embedding_model(self.dec_inputs)
            targets_emb = self.embedding_model(self.targets)

            # pre-train
            g_outputs_pre, g_state_pre, _ = self.define_generator(
                inputs_emb, pretrain=True)
            self.g_logits_pre, _, _, _ = self.language_model(g_outputs_pre)

            # train.
            g_outputs, g_state, g_logprobs = self.define_generator(
                inputs_emb, pretrain=False)
            self.g_outputs = g_outputs
            self.gen_state = g_state

            # only use it for prediction.
            probs = tf.transpose(g_logprobs.stack(), perm=[1, 0, 2])
            prob = tf.reshape(probs, [-1, self.loader.vocab_size])
            prob = tf.reshape(probs, [-1, self.loader.vocab_size])[-1]

            self.g_top_values, self.g_top_indexs = tf.nn.top_k(
                prob, k=self.para.BEAM_SEARCH_SIZE, sorted=True)

        with tf.variable_scope('discriminator'):
            log('define discriminator.')

            self.logits_D_real, self.D_real = \
                self.define_discriminator(targets_emb)
            self.logits_D_fake, self.D_fake = \
                self.define_discriminator(g_outputs, reuse=True)
            self.gradient_penalty = self.do_gradient_penalty(targets_emb,
                                                             g_outputs)

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
                    self.g_logits_pre,
                    [self.para.BATCH_SIZE, self.sent_len,
                     self.loader.vocab_size]),
                targets=self.targets,
                weights=tf.cast(self.targets, tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)

        with tf.name_scope("train_loss"):
            loss_G = - tf.reduce_mean(self.logits_D_fake)
            loss_D = - tf.reduce_mean(
                self.logits_D_real - self.logits_D_fake)
            loss_D += self.para.WGAN_GRADIENT_PENALTY * self.gradient_penalty
            self.loss_G, self.loss_D = loss_G, loss_D

    """model running part."""
    def running(self):
        log('------ pretraining ------ \n')
        for c_epoch in range(self.epoch_pointer.eval(),
                             self.para.EPOCH_PRETRAIN):

            log('pretrain epoch {}'.format(c_epoch))
            self.pretrain_step()

            if c_epoch % self.para.CHECKPOINT_EVERY == 0:
                self.save_model()

        log('------ training ------ \n')
        upper_epoches = self.para.EPOCH_PRETRAIN + self.para.EPOCH_TRAIN

        for c_epoch in range(self.epoch_pointer.eval() + 1, upper_epoches):

            log('train epoch {}'.format(c_epoch))
            self.train_step(c_epoch)

            if c_epoch % self.para.CHECKPOINT_EVERY == 0:
                self.save_model()

    def pretrain_step(self):
        start_epoch_time = datetime.datetime.now()
        losses = {'losses_D': [], 'losses_G': []}
        cur_batch = 0

        while True:
            data, noise, index = self.loader.next_batch()
            if data is None:
                break

            # load data.
            _, inp, tar = self.loader.process_bucket(data, index)

            feed_dict_D_pos = {
                self.targets: inp,
                self.target_labels: np.ones((self.para.BATCH_SIZE, 1),
                                            dtype=np.float32),
                self.sent_len: inp.shape[1],
                self.dropout: self.para.DROPOUT_RATE
            }
            feed_dict_D_neg = {
                self.targets: self.loader.swap_random_pos(inp),
                self.target_labels: np.zeros((self.para.BATCH_SIZE, 1),
                                             dtype=np.float32),
                self.sent_len: inp.shape[1],
                self.dropout: self.para.DROPOUT_RATE
            }
            feed_dict_G = {
                self.dec_inputs: inp,
                self.targets: tar,
                self.sent_len: inp.shape[1],
                self.dropout: self.para.DROPOUT_RATE
            }

            # train D.
            _, loss_D_pos = self.sess.run(
                [self.op_pretrain_D, self.loss_D_pre], feed_dict_D_pos)
            _, loss_D_neg = self.sess.run(
                [self.op_pretrain_D, self.loss_D_pre], feed_dict_D_neg)

            # train G.
            _, loss_G = self.sess.run(
                [self.op_pretrain_G, self.loss_G_pre], feed_dict_G)

            # record loss.
            losses['losses_D'].append(loss_D_pos + loss_D_neg)
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
                _, loss_D = self.sess.run(
                    [self.op_train_D, self.loss_D], feed_dict_D)
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

    """sentence generation part."""
    def sample_from_latent_space(self, enc_inp):
        """sample answer from the latent space."""
        log('generate sentence from latent space.')
        # init.
        sequence = [2]
        dec_inp = self.build_sample_input(sequence)
        soft_argmax = map(int, self.para.SOFT_ARGMAX.split(','))[1]

        candidates = []
        options = []

        feed_dict = {
            self.dec_inputs: dec_inp, self.dropout: 1.0,
            self.sent_len: 1,
            self.soft_argmax: soft_argmax
        }

        values, indices, state = self.sess.run(
            [self.g_top_values, self.g_top_indexs, self.gen_state],
            feed_dict)

        for i in range(len(values)):
            candidates.append([values[i], [indices[i]]])

        # start to sample.
        best_sequence = None
        highest_score = -sys.maxint - 1

        while True:
            for i in range(len(candidates)):
                sequence = candidates[i][1]
                score = candidates[i][0]

                # if sequence end, evaluate.
                if sequence[-1] == 3 \
                        or len(sequence) >= self.para.SAMPLING_LENGTH:
                    if score > highest_score:
                        highest_score = score
                        best_sequence = sequence
                    continue

                # if not, continue searching.
                dec_inp = self.build_sample_input(sequence)

                feed_dict = {
                    self.gen_init_state: state,
                    self.dec_inputs: dec_inp, self.dropout: 1.0,
                    self.sent_len: 1,
                    self.soft_argmax: soft_argmax
                }
                values, indices = self.sess.run(
                    [self.g_top_values, self.g_top_indexs], feed_dict)

                for j in range(len(values)):
                    new_sequence = list(sequence)
                    new_sequence.append(indices[j])
                    options.append([score + values[j], new_sequence])

            options.sort(reverse=True)
            candidates = []

            for i in range(min(len(options), self.para.BEAM_SEARCH_SIZE)):
                if options[i][0] > highest_score:
                    candidates.append(options[i])

            options = []
            if len(candidates) == 0:
                break

        return best_sequence[:-1]

    def get_test_sample(self):
        random_index = np.random.randint(1, len(self.loader.lines))

        for _ in range(random_index):
            data, noise, index = self.loader.next_batch()

        enc_inp, dec_inp, dec_tar = self.loader.process_bucket(data, index)
        return enc_inp, dec_inp, dec_tar, data

    def print_out_comparison(self, ckpt):
        log('restore model from {}'.format(ckpt.model_checkpoint_path))

        # test.
        _, dec_inp, dec_tar, data = self.get_test_sample()
        faked = self.sample_from_latent_space(dec_inp)

        log('true sentence: {}'.format(
            ' '.join([str(self.loader.inv_dict[x]) for x in data[0]])))
        log('faked sentence: {}'.format(
            ' '.join([str(self.loader.inv_dict[x]) for x in faked])))
