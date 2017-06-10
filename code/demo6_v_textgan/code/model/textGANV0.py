# -*- coding: utf-8 -*-
import sys
import datetime
from os.path import join

import numpy as np
import tensorflow as tf

from code.utils.logger import log
from code.core.inferenceModel import InferenceModel


class TextGANV0(InferenceModel):
    """The textG here simply follow the paper.
    It uses an RNN network as the generator.
    """

    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(TextGANV0, self).__init__(para, loader, sess, infer)

    """model definition part."""
    def define_enc_dec(self):
        """define the seq2seq model."""
        # define encoder.
        self.enc_lstm = self.lstm(self.para)
        enc_cell = self.enc_lstm.inherit_lstm_fn_from_tf(
            self.para.RNN_TYPE, self.para.RNN_SIZE,
            self.para.PROJECTION_SIZE, self.dropout)
        self.enc_init_state = enc_cell.zero_state(self.para.BATCH_SIZE,
                                                  tf.float32)
        self.dec_lstm = self.lstm(self.para)
        dec_cell = self.dec_lstm.inherit_lstm_fn_from_tf(
            self.para.RNN_TYPE, self.para.RNN_SIZE,
            self.para.PROJECTION_SIZE, self.dropout)

        _, enc_state = tf.nn.dynamic_rnn(
            cell=enc_cell,
            inputs=self.enc_inputs_emb,
            initial_state=self.enc_init_state,
            dtype=tf.float32,
            time_major=False, scope='encoder')

        self.dec_init_state = enc_state

        dec_outputs, dec_states = tf.nn.dynamic_rnn(
            cell=dec_cell,
            inputs=self.dec_inputs_emb,
            initial_state=self.dec_init_state,
            dtype=tf.float32,
            time_major=False, scope='decoder')
        self.dec_final_state = dec_states
        self.dec_outputs = dec_outputs

    def define_inference(self):
        """define inference procedure."""
        # init some functions.
        self.enc_inputs_emb = self.embedding_model(self.enc_inputs)
        self.dec_inputs_emb = self.embedding_model(self.dec_inputs)

        # inference.
        self.define_enc_dec()

        # get outputs.
        logits, _, log_probs, _ = self.language_model(self.dec_outputs)

        # prediction.
        self.logits, self.log_probs = logits, log_probs

        logit = log_probs[-1]
        self.top_values, self.top_indexs = tf.nn.top_k(
            logit, k=self.para.BEAM_SEARCH_SIZE, sorted=True)

    def define_loss(self):
        """define the loss."""
        with tf.name_scope("loss"):
            flat_targets = tf.reshape(self.targets, [-1])
            total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.log_probs, labels=flat_targets)
            self.loss = tf.reduce_mean(total_loss)

    """model running part."""
    def running(self):
        best_loss = float('inf')

        log('------ pretraining ------ \n')
        for c_epoch in range(self.epoch_pointer.eval(),
                             self.para.EPOCH_PRETRAIN):

            log('train epoch {}'.format(c_epoch))
            avg_loss = self.train_step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model()

    def train_step(self):
        losses = []
        start_epoch_time = datetime.datetime.now()
        cur_batch = 0

        while True:
            data, index = self.loader.next_batch()
            if data is None:
                break

            enc_inp, dec_inp, dec_tar = self.loader.process_bucket(data, index)

            feed_dict = {
                self.enc_inputs: enc_inp,
                self.dec_inputs: dec_inp,
                self.targets: dec_tar,
                self.dropout: self.para.DROPOUT_RATE}
            _, loss, summary = self.sess.run(
                [self.op_train, self.loss, self.op_train_summary], feed_dict)

            cur_batch += 1
            losses.append(loss)
            self.train_summary_writer.add_summary(summary)

            sys.stdout.write('\rcurrent batch: {}, mean loss = {}'.format(
                cur_batch, np.mean(losses)))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
        end_epoch_time = datetime.datetime.now()
        speed = 1.0 * (end_epoch_time - start_epoch_time).seconds / cur_batch

        log('loss: {}, speed: {:.2f} seconds/batch.'.format(
            np.mean(losses), speed))
        return np.mean(losses)

    def define_train_op(self):
        # get the training vars for both networks
        vars_all = tf.trainable_variables()

        # define optimizer
        optimizer = self.define_optimizer(self.para.LEARNING_RATE_G)

        # define pretrain op
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_all), self.para.GRAD_CLIP)
        self.grads_and_vars = zip(grads, vars_all)
        self.op_train = optimizer.apply_gradients(self.grads_and_vars)

    def keep_tracking(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged = self.keep_tracking_grad_and_vals(
            self.grads_and_vars)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar('loss', self.loss)

        # Summaries
        self.op_train_summary = tf.summary.merge(
            [loss_summary, grad_summaries_merged])

        # log("writing to {}\n".format(self.out_dir))
        train_summary_dir = join(self.out_dir, "summaries", "train")

        self.train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, self.sess.graph)

    """sentence generation part."""
    def sample_from_latent_space(self, enc_inp):
        """sample answer from the latent space."""
        log('generate sentence from latent space.')
        # init.
        sequence = [2]
        dec_inp = self.build_sample_input(sequence)

        candidates = []
        options = []

        feed_dict = {
            self.enc_inputs: enc_inp,
            self.dec_inputs: dec_inp, self.dropout: 1.0
        }

        values, indices, state = self.sess.run(
            [self.top_values, self.top_indexs, self.dec_final_state],
            feed_dict)

        for i in range(len(values)):
            candidates.append([values[i], [indices[i]]])

        # start to sample.
        best_sequence = None
        highest_score = - sys.maxint - 1

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
                    self.dec_init_state: state,
                    self.dec_inputs: dec_inp, self.dropout: 1.0}
                values, indices = self.sess.run(
                    [self.top_values, self.top_indexs], feed_dict)

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
            data, index = self.loader.next_batch()

        enc_inp, dec_inp, dec_tar = self.loader.process_bucket(data, index)
        return enc_inp, dec_inp, dec_tar, data

    def print_out_comparison(self, ckpt):
        log('restore model from {}'.format(ckpt.model_checkpoint_path))

        # test.
        enc_input, _, _, data = self.get_test_sample()
        faked = self.sample_from_latent_space(enc_input)

        log('true question: {}'.format(
            ' '.join([str(self.loader.inv_dict[x]) for x in data[0][0]])))
        log('true answer: {}'.format(
            ' '.join([str(self.loader.inv_dict[x]) for x in data[0][1]])))
        log('faked answer: {}'.format(
            ' '.join([str(self.loader.inv_dict[x]) for x in faked])))
