# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from code.utils.logger import log
from code.core.lstm import LSTM
from code.core.basicModel import BasicModel
from code.core.wordSearch import WordSearch


class InferenceModel(BasicModel):
    def __init__(self, para, loader, sess, infer=False):
        """init parameters."""
        super(InferenceModel, self).__init__(para, loader, sess, infer)

        # init the basic model..
        self.define_placeholder()
        self.define_pointer()
        self.lstm = LSTM

    def prepare_generator(self):
        self.lstm_G_enc = self.lstm(self.para.RNN_SIZE, self.para.BATCH_SIZE)
        self.cell_G_enc = self.lstm_G_enc.cell
        self.cell_G_enc_init_state = self.lstm_G_enc.init_state()

        self.lstm_G_dec = self.lstm(self.para.RNN_SIZE, self.para.BATCH_SIZE)
        self.cell_G_dec = self.lstm_G_dec.cell
        self.cell_G_dec_init_state = None

    def define_generator(self, z=None, embedded_x=None, reuse=False):
        # inference for the generator.
        embedded_x_enc, embedded_x_dec = embedded_x

        # encoder.
        cell_type = self.lstm_G_enc.standard_lstm_unit
        enc_state = self.cell_G_enc_init_state

        with tf.variable_scope('rnn_enc') as scope_rnn:
            for time_step in range(self.enc_length):
                if time_step > 0 or reuse:
                    scope_rnn.reuse_variables()
                input_enc = embedded_x_enc[:, time_step, :]
                _, enc_state = self.cell_G_enc(
                    cell_type, input_enc, enc_state, None)

        self.cell_G_dec_init_state = enc_state
        dec_state = self.cell_G_dec_init_state

        # decoder.
        if reuse:
            input_dec = z
        else:
            input_dec = embedded_x_dec[:, 0, :]

        embeddings, logits, probs, words = [], [], [], []

        for time_step in range(self.dec_length):
            with tf.variable_scope('rnn_dec') as scope_rnn:
                if time_step > 0 or reuse:
                    scope_rnn.reuse_variables()
                if (time_step > 1 and reuse) or self.infer:
                    cell_type = self.lstm_G_dec.noise_lstm_unit
                else:
                    cell_type = self.lstm_G_dec.standard_lstm_unit

                cell_output, dec_state = self.cell_G_dec(
                    cell_type, input_dec, dec_state, z)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, word = self.language_model(
                cell_output, reuse=True)

            if reuse:
                input_dec = self.get_approx_embedding(soft_prob)
            else:
                input_dec = embedded_x_dec[:, time_step + 1, :] \
                    if time_step < self.dec_length - 1 else None

            # save the middle result.
            embeddings.append(cell_output)
            logits.append(logit)
            probs.append(prob)
            words.append(word)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.dec_length, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.dec_length, self.loader.vocab_size])
        embeddings = tf.reshape(
            tf.concat(embeddings, 0),
            [-1, self.dec_length, self.para.EMBEDDING_SIZE])
        return embeddings, logits, probs, words, enc_state, dec_state

    def define_discriminator(self, embedded, reuse=False):
        """define the discriminator."""
        if reuse:
            tf.get_variable_scope().reuse_variables()

        input = tf.expand_dims(embedded, 3)

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
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                h = self.leakyrelu(conv, b, alpha=1.0/5.5)
                pooled = self.max_pool(
                    h,
                    ksize=[
                        1,
                        self.loader.sentence_length - 1 - conv_spatial + 1,
                        1,
                        1],
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                pooled_outputs.append(pooled)

        num_filters_total = sum([x[1] for x in archits])
        h_pool_flat = tf.reshape(
            tf.concat(pooled_outputs, 3), [-1, num_filters_total])

        # output the classification result.
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.para.LABEL_DIM],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                'b',
                shape=[self.para.LABEL_DIM],
                initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.nn.xw_plus_b(h_pool_flat, W, b, name="scores")
        return logits, tf.nn.sigmoid(logits)

    def define_inference(self):
        """"define the inference procedure in training phase."""
        with tf.variable_scope('generator'):
            self.embedding_model()
            self.language_model()
            self.prepare_generator()
            embedded_x_enc = self.embedding_model(self.x_enc, reuse=True)
            embedded_x_dec = self.embedding_model(self.x_dec, reuse=True)
            embedded_y = self.embedding_model(self.y, reuse=True)
            embedded_x = (embedded_x_enc, embedded_x_dec)

            _, self.logits_G_pretrain, _, _, _, _ = self.define_generator(
                embedded_x=embedded_x)

            self.embedded_G, self.logits_G, self.probs_G, \
                self.outputs_G, self.enc_state_G, self.dec_state_G = \
                self.define_generator(
                    z=self.z, embedded_x=embedded_x, reuse=True)

        with tf.variable_scope('discriminator'):
            self.logits_D_real, self.D_real = self.define_discriminator(
                embedded_y)
            self.logits_D_fake, self.D_fake = self.define_discriminator(
                self.embedded_G, reuse=True)

    def define_pretrain_loss(self):
        """define the pretrain loss.

        For `sigmoid_cross_entropy_with_logits`, where z is label, x is data.
        we have z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x)).
        """
        with tf.name_scope("pretrain_loss"):
            # deal with discriminator.
            self.loss_D_pretrain = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_D_real,
                    labels=self.y_label
                )
            )

            self.loss_G_pretrain = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits_G_pretrain,
                targets=self.y,
                weights=self.ymask,
                average_across_timesteps=True,
                average_across_batch=True)

    """define method related to language model/word embedding."""
    def language_model(self, output=None, reuse=False):
        with tf.variable_scope('lm') as scope:
            if not reuse:
                softmax_w = tf.get_variable(
                    "softmax_w",
                    shape=[self.para.RNN_SIZE, self.loader.vocab_size],
                    initializer=tf.contrib.layers.xavier_initializer())
                softmax_b = tf.get_variable(
                    "softmax_b",
                    shape=[self.loader.vocab_size],
                    initializer=tf.contrib.layers.xavier_initializer())
            elif reuse and output is not None:
                scope.reuse_variables()
                softmax_w = tf.get_variable("softmax_w")
                softmax_b = tf.get_variable("softmax_b")

                logit = tf.matmul(output, softmax_w) + softmax_b
                prob = tf.nn.softmax(logit)
                soft_prob = tf.nn.softmax(logit * self.soft_argmax)
                output = tf.stop_gradient(tf.argmax(prob, 1))
                return logit, prob, soft_prob, output
            else:
                raise 'invaild usage.'

    def embedding_model(self, words=None, reuse=False):
        """word embedding."""
        with tf.variable_scope("embedding") as scope:
            if not reuse:
                # learn embedding during the training
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        "embedding",
                        shape=[
                            self.loader.vocab_size, self.para.EMBEDDING_SIZE],
                        initializer=tf.contrib.layers.xavier_initializer())
            else:
                scope.reuse_variables()
                embedding = tf.get_variable("embedding")
                embedded_words = tf.nn.embedding_lookup(embedding, words)
                return embedded_words

    def get_approx_embedding(self, soft_prob):
        with tf.variable_scope("embedding") as scope:
            scope.reuse_variables()
            embedding = tf.get_variable("embedding")
        return tf.matmul(soft_prob, embedding)

    # add function to generate sentence.
    def sample_from_latent_space(self, question, max_length=30):
        """generate sentence from latent space.."""
        # init.
        state = self.sess.run(self.cell_G_enc_init_state)
        state = self.sess.run(
            [self.enc_state_G], feed_dict={
             self.x_enc: [question],
             self.cell_G_enc_init_state: state}
        )

        log('generate sentence. beam search:{}.'.format(self.para.BEAM_SEARCH))
        word = 'go'
        generated_sentence = []
        word_search = WordSearch(self.loader.vocab, self.para)
        z = np.random.uniform(size=(1, self.para.EMBEDDING_SIZE))

        log('...decide sampling type.')
        if self.para.SAMPLING_TYPE == 'argmax':
            basic_sampler = np.argmax
        else:
            basic_sampler = word_search.weighted_pick

        log('...start sampling.')
        while True:
            input = np.zeros((1, 1))
            input[0, 0] = self.loader.vocab.get(word, 0)
            [probs, state] = self.sess.run(
                [self.probs_G, self.dec_state_G], feed_dict={
                    self.z: z, self.x_dec: input,
                    self.cell_G_dec_init_state: state
                }
            )
            sample_index = basic_sampler(probs[0, 0])
            generated_sentence.append(sample_index)
            if word == 'eos' or len(generated_sentence) > max_length:
                break
        return generated_sentence
