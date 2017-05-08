from generator import Generator as G
from dataloader import g_data_loader as G_ld

import numpy as np
import tensorflow as tf
import random
import helpers

import fuse.fuse_utils as fu
import fuse.fusing as fusing
import fuse.fuser as fuser


EMB_DIM = 256
HID_DIM = 64
SEQ_LEN = 20
START_TOKEN = 0
PRE_EPOCH_NUM = 10
SEED = 1234
BATCH_SIZE = 1
VOCAB_SIZE = 20525
LR = 0.01

data_path = "./datasets/bbt_concate.txt"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    G_dataloader = G_ld(BATCH_SIZE, 20, data_path)
    FUSER = fuser.Fuser()
    generator = G(FUSER, VOCAB_SIZE, BATCH_SIZE, EMB_DIM, HID_DIM, SEQ_LEN, START_TOKEN, LR)
    print "Finish building the fuser and the generator..."

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print "Finish initializing the session..."

    # generator.fuser.test_feed(sess)
    # log = open("save/experiment-log.txt", "w")
    # log.write("Pretraining the generator...")
    G_dataloader.create_batches()
    for epoch in xrange(PRE_EPOCH_NUM):
        print "This is epoch %d: " % epoch
        loss = helpers.pre_train_epoch(sess, generator, G_dataloader)
        print loss

if __name__ == "__main__":
    main()
