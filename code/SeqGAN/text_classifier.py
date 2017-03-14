import tensorflow as tf

def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    ''' highway network http://arxiv.org/abs/1505.00387 '''
    output = input_
    for idx in xrange(layer_size):
        output = f(tf.nn.rnn_cell._linear(input_, size, 0, scope='output_lin_%d' % idx))
        transform_gate = tf.sigmoid(
            tf.nn.rnn_cell._linear(input_, size, 0,
            scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_

    return output

class TextCNN(object):
    '''
        CNN for text classification
        Embedding layer + conv + max_pooling + softmax
    '''
    def __init__(self, sequence_len, num_classes, vocab_size, emb_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # tracking the l2 regularization loss
        l2_loss = tf.constant(0.)

        # Embedding
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, emb_size], -1., 1.),
                name="W")
            # print W
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # print self.embedded_chars
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # print self.embedded_chars_expanded

        # create convolution layers, for each filer depth, connected with a
        # max_pooling. zip(filter_sizes, num_filters)
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.name_scope("conv-max_pool-%s" % filter_size):
                filter_shape = [filter_size, emb_size, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_len-filter_size+1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool"
                )

                pooled_outputs.append(pooled)

        tot_num_filters = sum(num_filters)
        self.h_pool = tf.concat(axis=3, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, tot_num_filters])

        # with tf.name_scope("highway"):
        #     self.h_highway = highway(self.h_pool_flat,
        #     self.h_pool_flat.get_shape()[1], 1, 0)

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([tot_num_filters,
                                            num_classes], stddev=0.1, name="W"))
            b = tf.Variable(tf.constant(.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                                        logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")


if __name__ == "__main__":
    classifier = TextCNN(10, 2, 5000, 36, 15, 5)
