from tensorflow.python.ops import tensor_array_ops, control_flow_ops
# from overrides import overrides

import tensorflow as tf

class LSTM(object):
    def __init__(self, num_emb, batch_size, emb_dim,
                hidden_dim, sequence_len, start_token,
                learning_rate=0.01, rwd_gamma=0.95):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_len = sequence_len
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.rwd_gamma = rwd_gamma
        self.g_params = []
        self.d_params = []
        self.temperature = 1.
        self.grad_clip = 5.
        self.exp_rwd = tf.Variable(tf.zeros([self.sequence_len]))

        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            # recurrent unit, from h_t_minus_1 to h_t
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)
            # output unit, from h_t to o_t
            self.g_output_unit = self.create_output_unit(self.g_params)

        # input placeholder, sequences are index of true data, not include start token
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_len])

        # placeholder for rwd from D or rollout policy
        self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.sequence_len])

        # processing input batch
        with tf.device("/cpu:0"):
            inputs = tf.split(axis=1, num_or_size_splits=self.sequence_len, value=tf.nn.embedding_lookup(self.g_embeddings, self.x))
            self.processed_x = tf.stack([tf.squeeze(input_, [1]) for input_ in inputs])

        # initialize hidden states
        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_len, dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_len, dynamic_size=False, infer_shape=True)

        def _g_recurrence(i, x_t, h_t_minus_1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_t_minus_1)
            o_t = self.g_output_unit(h_t) # here are logists
            log_prob = tf.log(tf.nn.softmax(o_t)) # map to prob
            ### come back to check dimensionality ###

            # for each element in batch, find next token
            # next token, index
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)

            # batch_size x emb_dim
            x_t_plus_1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)

            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_emb, 1.0, 0.0), tf.nn.softmax(o_t)), 1))
            gen_x = gen_x.write(i, next_token) # index, next_token of each batch element

            return i+1, x_t_plus_1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_len,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
            tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, gen_o, gen_x))

        self.gen_x = self.gen_x.stack()
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0]) # batch_size x sequence_len
        # supervised pre train
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_len, dynamic_size=False, infer_shape=True)

        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_len, dynamic_size=False, infer_shape=True)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        def _pretrain_recurrence(i, x_t, h_t_minus_1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_t_minus_1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t)) #batch x vocab_size
            x_t_plus_1 = ta_emb_x.read(i)
            return i+1, x_t_plus_1, h_t, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_len,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, g_predictions))

        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])

        # pretrain loss
        self.pretrain_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
            )
        )/(self.sequence_len * self.batch_size)

        # update pretrain
        pretrain_opt = self.g_optimizer(self.learning_rate)

        self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.pretrain_loss, self.g_params), self.grad_clip)

        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

        ###########################
        ## unsupervised training

        self.g_loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
                ), 1) * tf.reshape(self.rewards, [-1])
            )

        g_opt = self.g_optimizer(self.learning_rate)

        self.g_grad, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params), self.grad_clip)

        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))


    def generate(self, session):
        outputs = session.run([self.gen_x])
        return outputs[0]

    def pretrain_step(self, session, x):
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions], feed_dict={self.x: x})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def g_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

    def create_recurrent_unit(self, params):
        # creat recurrent units, three gates i, f, o, c_
        # input, forget, output then update memory cell
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))

        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_t_m_1):
            prev_hidden_state, c_prev = tf.unstack(hidden_memory_t_m_1)

            # Input gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(prev_hidden_state, self.Ui) + self.bi
            )

            # Forget gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(prev_hidden_state, self.Uf) + self.bf
            )

            # Output gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(prev_hidden_state, self.Uog) + self.bog
            )

            # New memo cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(prev_hidden_state, self.Uc) + self.bc
            )

            # refresh memo cell
            c = f * c_prev + i * c_

            # current hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(self.init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memo_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memo_tuple)
            # hidden_state : batch_size x hidden_dim
            logists = tf.matmul(hidden_state, self.Wo) + self.bo
            # output tf.nn.softmax(logists)
            return logists

        return unit
