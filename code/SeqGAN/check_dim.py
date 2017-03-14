import tensorflow as tf
import numpy as np

params1 = tf.constant(np.random.rand(3, 4))
params2 = tf.random_uniform([500, 10], -1.0, 1.0)
ids = tf.constant([0, 1])
output = tf.nn.embedding_lookup(params2, ids)

with tf.Session() as sess:
    print "params", sess.run(params2).shape
    print sess.run(output).shape
