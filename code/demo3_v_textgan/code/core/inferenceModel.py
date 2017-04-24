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

    def define_generator(self, z=None, embedded_x=None):
        self.lstm_G = self.lstm(self.para.RNN_SIZE, self.para.BATCH_SIZE)
        self.cell_G = self.lstm_G.cell

        embeddings, logits, probs, words = [], [], [], []

        if z is not None and embedded_x is None:
            log("enter GAN's generator mode.")
            input = z
            state = self.lstm_G.init_state()
            gan_mode = True
        elif z is None and embedded_x is not None:
            log("enter standard seq2seq's generator mode.")
            input = embedded_x[:, 0, :]
            state = self.lstm_G.init_state()
            gan_mode = False
        else:
            raise 'invaild inference procedure.'

        for time_step in range(self.para.SENTENCE_LENGTH):
            with tf.variable_scope('rnn') as scope_rnn:
                scope_rnn.reuse_variables()
                if time_step > 1 and gan_mode:
                    cell_type = self.lstm_G.noise_lstm_unit
                else:
                    cell_type = self.lstm_G.standard_lstm_unit
                cell_output, state = self.cell_G(cell_type, input, state, z)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, word = self.language_model(
                cell_output, reuse=True)

            if gan_mode:
                input = self.get_approx_embedding(soft_prob)
            else:
                input = embedded_x[:, time_step + 1, :] \
                    if time_step < self.para.SENTENCE_LENGTH - 1 else None

            # save the middle result.
            embeddings.append(cell_output)
            logits.append(logit)
            probs.append(prob)
            words.append(word)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.para.SENTENCE_LENGTH, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.para.SENTENCE_LENGTH, self.loader.vocab_size])
        embeddings = tf.reshape(
            tf.concat(embeddings, 0),
            [-1, self.para.SENTENCE_LENGTH, self.para.EMBEDDING_SIZE])
        return embeddings, logits, probs, words, state

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
                        self.loader.sentence_length - conv_spatial + 1,
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
            embedded_x = self.embedding_model(self.x, reuse=True)
            embedded_y = self.embedding_model(self.y, reuse=True)

            _, self.logits_G_pretrain, _, _, _ = self.define_generator(
                embedded_x=embedded_x)

            self.embedded_G, self.logits_G, self.probs_G, \
                self.outputs_G, self.state_G = \
                self.define_generator(z=self.z)

            self.top_value, self.top_index = tf.nn.top_k(
                self.logits_G, k=self.para.BEAM_SEARCH_SIZE, sorted=True)

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
    def sample_from_latent_space(self, num=200, prime='go'):
        """generate sentence from latent space.."""
        # init.
        z = np.random.uniform(size=(1, self.para.EMBEDDING_SIZE * 2))
        state = tf.contrib.rnn.LSTMStateTuple(
                c=z[:, :self.para.EMBEDDING_SIZE],
                h=z[:, self.para.EMBEDDING_SIZE:])

        log('generate sentence. beam search:{}.'.format(self.para.BEAM_SEARCH))
        generated_sentence = prime.split()
        word_search = WordSearch(self.loader.vocab, self.para)

        word = 'go'
        if self.para.BEAM_SEARCH:
            # feed `go` into the generator.
            input = np.zeros((1, 1))
            input[0, 0] = self.loader.vocab.get(word, 0)
            feed = {self.x: input, self.cell_G_init_state: state}
            probs, state, values, indices = self.sess.run(
                [self.probs_G, self.state_G,
                 self.top_value, self.top_index],
                feed)

            values, indices = values[0][0], indices[0][0]
            for i in range(len(values)):
                word_search.beam_candidates.append((values[i], [indices[i]]))

        log('...decide sampling type.')
        if self.para.SAMPLING_TYPE == 'argmax':
            basic_sampler = np.argmax
        else:
            basic_sampler = word_search.weighted_pick

        log('...start sampling.')
        for n in range(num):
            if not self.para.BEAM_SEARCH:
                input = np.zeros((1, 1))
                input[0, 0] = self.loader.vocab.get(word, 0)
                feed = {self.x: input, self.cell_G_init_state: state}
                [probs, state] = self.sess.run(
                    [self.probs_G, self.state_G], feed)
                sample_index = basic_sampler(probs[0, 0])
                word = self.loader.words[sample_index]
                generated_sentence.append(word)
                if word == 'eos':
                    break
            else:
                word_search.beam_search(
                    self.sess, self.x, self.cell_G_init_state,
                    self.top_value, self.top_index, state)

                generated_sentence = [
                    self.loader.words[s] for s in word_search.best_sequence]

        generated_sentence = ' '.join(generated_sentence)
        return generated_sentence
