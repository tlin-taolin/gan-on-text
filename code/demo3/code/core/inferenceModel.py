# -*- coding: utf-8 -*-
import tensorflow as tf

from code.core.basicModel import BasicModel
from code.core.lstm import LSTM


class InferenceModel(BasicModel):
    """Define some infernece components for ease of modification."""
    def __init__(self, para, loader, sess, training):
        """init."""
        super(InferenceModel, self).__init__(para, loader, sess, training)

    """define inference main entry."""
    def define_inference(self, generator, discriminator):
        """"define the inference procedure in training phase."""
        with tf.variable_scope('generator'):
            self.embedding()
            self.language_model()

            self.pre_logit, self.pre_prob, self.pre_out, _, self.pre_state\
                = generator(x=self.x, pretrain=True)
            self.G_logit, self.G_prob, self.G_out, self.G_embedded_out, self.G_state\
                = generator(z=self.z, pretrain=False)
            embedded_x = self.embedding(self.x, reuse=True)

            self.pre_top_value, self.pre_top_index = tf.nn.top_k(
                self.pre_logit, k=self.para.BEAM_SEARCH_SIZE, sorted=True)
            self.G_top_value, self.G_top_index = tf.nn.top_k(
                self.G_logit, k=self.para.BEAM_SEARCH_SIZE, sorted=True)

        if self.training:
            with tf.variable_scope('discriminator') as discriminator_scope:
                self.logit_D_real, self.D_real \
                    = discriminator(embedded_x, discriminator_scope)

                # get discriminator on fake data. the reuse=True, which
                # specifies we reuse the discriminator ops for new placeholder.
                self.logit_D_fake, self.D_fake \
                    = discriminator(self.G_embedded_out, discriminator_scope,
                                    reuse=True)

    """define inference components."""
    def define_discriminator_as_CNN(self, embedded, scope, reuse=False):
        """define the discriminator."""
        if reuse:
            scope.reuse_variables()

        input = tf.expand_dims(embedded, 3)

        # build a series for 'CONV + RELU + POOL' architecture.
        archits = zip(self.para.D_CONV_SPATIALS, self.para.D_CONV_DEPTHS)
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

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_val)

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
            logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        return logits, tf.nn.sigmoid(logits)

    def define_discriminator_as_LSTM(self, embedded, dis_scope, reuse=False):
        """define the discriminator."""
        state = self.D_cell_init_state

        with tf.variable_scope('rnn') as scope_rnn:
            for time_step in range(self.loader.sentence_length):
                if time_step > 0 or reuse:
                    scope_rnn.reuse_variables()
                output, state = self.D_cell(embedded[:, time_step, :], state)

        # output the classification result.
        with tf.variable_scope("output") as scope:
            if not reuse:
                W = tf.get_variable(
                    "W",
                    shape=[self.para.RNN_SIZE, self.para.LABEL_DIM],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(
                    'b',
                    shape=[self.para.LABEL_DIM],
                    initializer=tf.contrib.layers.xavier_initializer())
            else:
                scope.reuse_variables()
                W = tf.get_variable("W")
                b = tf.get_variable("b")
            logits = tf.nn.xw_plus_b(output, W, b)
        return logits, tf.nn.sigmoid(logits)

    def define_generator_as_LSTM(self, z=None, x=None, pretrain=False):
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            input = z

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()
                cell_output, state = self.G_cell(input, state)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = self.get_approx_embedding(soft_prob)

            # save the middle result.
            logits.append(logit)
            probs.append(prob)
            outputs.append(output)
            embedded_outputs.append(input)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        embedded_outputs = tf.reshape(
            tf.concat(embedded_outputs, 0),
            [-1, self.loader.sentence_length, self.para.EMBEDDING_SIZE])
        return logits, probs, outputs, embedded_outputs, state

    def define_generator_as_noiseLSTM(self, z=None, x=None, pretrain=False):
        """define the generator.

        feed z to every lstm cells to guide the sentence generation.
        """
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            cell_type = self.lstm.standard_lstm_unit
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            cell_type = self.lstm.noise_lstm_unit
            input = z

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()
                if time_step == 0:
                    cell_output, state = self.lstm.cell(
                        self.lstm.standard_lstm_unit, input, state, z)
                else:
                    cell_output, state = self.lstm.cell(
                        cell_type, input, state, z)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = self.get_approx_embedding(soft_prob)

            # save the middle result.
            logits.append(logit)
            probs.append(prob)
            outputs.append(output)
            embedded_outputs.append(input)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        embedded_outputs = tf.reshape(
            tf.concat(embedded_outputs, 0),
            [-1, self.loader.sentence_length, self.para.EMBEDDING_SIZE])
        return logits, probs, outputs, embedded_outputs, state

    def define_generator_as_hiddenLSTM(self, z=None, x=None, pretrain=False):
        """define the generator.

        Feed z as an init state to the cell,
        and feed the embedding of '<go>' as the input of the first lstm cell.
        """
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            tmp = tf.ones(
                (self.para.BATCH_SIZE, 1),
                dtype=tf.int32) * self.loader.vocab['<go>']
            input = self.embedding(tmp, reuse=True)[:, 0, :]
            state = tf.contrib.rnn.LSTMStateTuple(c=z, h=z)

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()

                cell_output, state = self.G_cell(
                    self.lstm.standard_lstm_unit, input, state)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = self.get_approx_embedding(soft_prob)

            # save the middle result.
            logits.append(logit)
            probs.append(prob)
            outputs.append(output)
            embedded_outputs.append(input)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        embedded_outputs = tf.reshape(
            tf.concat(embedded_outputs, 0),
            [-1, self.loader.sentence_length, self.para.EMBEDDING_SIZE])
        return logits, probs, outputs, embedded_outputs, state

    def define_generator_as_hiddenLSTM_v(self, z=None, x=None, pretrain=False):
        """define the generator.

        Feed z as an init state to the cell,
        and feed the embedding of '<go>' as the input of the first lstm cell.
        Also note that we directly use the output
        """
        logits, probs, outputs, embedded_outputs = [], [], [], []
        state = self.G_cell_init_state

        if pretrain:
            inputs = self.embedding(x, reuse=True)
            input = inputs[:, 0, :]
        else:
            tmp = tf.ones(
                (self.para.BATCH_SIZE, 1),
                dtype=tf.int32) * self.loader.vocab['<go>']
            input = self.embedding(tmp, reuse=True)[:, 0, :]
            state = tf.contrib.rnn.LSTMStateTuple(c=z, h=z)

        for time_step in range(self.loader.sentence_length):
            with tf.variable_scope('rnn') as scope_rnn:
                if time_step > 0 or not pretrain:
                    scope_rnn.reuse_variables()

                cell_output, state = self.G_cell(
                    self.lstm.standard_lstm_unit, input, state)

            # feed the current cell output to a language model.
            logit, prob, soft_prob, output = self.language_model(
                cell_output, reuse=True)
            # decide the next input,either from inputs or from approx embedding
            if pretrain:
                input = inputs[:, time_step, :]
            else:
                input = cell_output
                embedded_output = self.get_approx_embedding(soft_prob)

            # save the middle result.
            logits.append(logit)
            probs.append(prob)
            outputs.append(output)
            embedded_outputs.append(embedded_output)

        logits = tf.reshape(
            tf.concat(logits, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        probs = tf.reshape(
            tf.concat(probs, 0),
            [-1, self.loader.sentence_length, self.loader.vocab_size])
        embedded_outputs = tf.reshape(
            tf.concat(embedded_outputs, 0),
            [-1, self.loader.sentence_length, self.para.EMBEDDING_SIZE])
        return logits, probs, outputs, embedded_outputs, state
