# from overrides import overrides

from gen_dataloader import Gen_Data_loader, Likelihood_Data_loader
from dis_dataloader import Dis_dataloader
from H_test import significance_test
from text_classifier import TextCNN
from target_lstm import TARGET_LSTM
from rollout import ROLLOUT

from pretrain import G_, get_trainable_model, generate_samples
from pretrain import target_loss, pre_train_epoch

import model
import numpy as np
import tensorflow as tf
import random
import time
import cPickle

SEQ_LEN = 20
START_TOKEN = 0
SEED = 10

# hyperparams for generator G
EBD_DIM = 16
HID_DIM = 16

PRE_EPC_NUM = 0
TRAIN_ITER = 1
BATCH_SIZE = 64

# Total number of batch
TOTOAL_BATCH = 800

# hyperparams for D
dis_embedding_dim = 16
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = .75
dis_l2_reg_lambda = 0.2

# training params
dis_batch_size = 64
dis_num_epochs = 3
dis_alter_epoch = 0

positive_file = 'save/real_data.txt'
negative_file = 'target_generate/generator_sample.txt'
eval_file = 'target_generate/eval_file.txt'

generated_num = 10000


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LEN)
    lik_data_loader = Likelihood_Data_loader(BATCH_SIZE, SEQ_LEN)
    dis_dataloader = Dis_dataloader()

    vocab_size = 5000

    best_score = 1000
    G = get_trainable_model(vocab_size)
    target_params = cPickle.load(open("save/target_params.pkl"))
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)

            # sequence_len, num_classes, vocab_size, emb_size,
            #  filter_sizes, num_filters, l2_reg_lambda=0.

    with tf.variable_scope("D"):
        cnn = TextCNN(
            sequence_len=SEQ_LEN,
            num_classes=2,
            vocab_size=vocab_size,
            emb_size=dis_embedding_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            l2_reg_lambda=dis_l2_reg_lambda
            )

    cnn_params = [param for param in tf.trainable_variables() if "D" in param.name]

    # training procedure for D
    dis_global_step = tf.Variable(0, name="global_step", trainable=False) #control D
    dis_optimizer = tf.train.AdamOptimizer(1e-4)
    dis_grads_and_vars = dis_optimizer.compute_gradients(cnn.loss, cnn_params, aggregation_method=2)
    dis_train_op = dis_optimizer.apply_gradients(dis_grads_and_vars, global_step=dis_global_step)

    # configure tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    generate_samples(sess, target_lstm, 64, 10000, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('log/experiment-log.txt', 'w')
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPC_NUM):
        print 'pre-train epoch:', epoch
        loss = pre_train_epoch(sess, G, gen_data_loader)
        if epoch % 5 == 0:
            generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
            lik_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, lik_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            connexion = str(epoch) + ' ' + str(test_loss) + '\n'
            log.write(connexion)

    generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
    lik_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, lik_data_loader)
    connexion = 'After pre-training:' + ' ' + str(test_loss) + '\n'
    log.write(connexion)

    generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
    lik_data_loader.create_batches(eval_file)
    significance_test(sess, target_lstm, lik_data_loader, 'significance/supervise.txt')

    print "finish pretrain..."
    print '''--------------- training the discriminator ------------'''
    for idx in xrange(dis_alter_epoch):
        print "epoch %d of training discriminator... "% idx
        generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
        dis_x_train, dis_y_train = dis_dataloader.load_train_data(positive_file, negative_file)
        dis_batches = dis_dataloader.batch_iteror(
                zip(dis_x_train, dis_y_train),
                dis_batch_size, dis_num_epochs)

        for batch in dis_batches:
            try:
                x_batch, y_batch = zip(*batch)
                feed = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, step = sess.run([dis_train_op, dis_global_step], feed)

            except ValueError:
                pass

    rollout = ROLLOUT(G, 0.8)

    print 'Training GAN: '
    log.write("Reinforcement training for generator... \n")
    for total_batch in xrange(TOTOAL_BATCH):
        print "Reinforcement training for generator... BATCH %d" %total_batch
        for it in xrange(TRAIN_ITER):
            samples = G.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, cnn)
            feed = {
                G.x: samples, G.rewards: rewards
            }
            _, g_loss = sess.run([G.g_updates, G.g_loss], feed_dict=feed)

            print sess.run(G.g_embeddings)

        if total_batch % 1 == 0 or total_batch == TOTOAL_BATCH-1:
            generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
            lik_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, lik_data_loader)
            connexion = str(total_batch) + " " + str(test_loss) + "\n"
            print "total_batch: ", total_batch, "test_loss: ", test_loss
            log.write(connexion)

            if test_loss < best_score:
                best_score = test_loss
                print "best score: ", best_score
                significance_test(sess, target_lstm, lik_data_loader, "significance/seqgan.txt")

        rollout.update_params()

        print "Tarining G"
        for idx_G in xrange(5):
            print "iter %d for training G" % idx_G
            generate_samples(sess, G, BATCH_SIZE, generated_num, negative_file)
            dis_x_train, dis_y_train = dis_dataloader.load_train_data(positive_file, negative_file)
            dis_batches = dis_dataloader.batch_iteror(zip(dis_x_train, dis_y_train), dis_batch_size, 3)

            for batch in dis_batches:
                try:
                    x_batch, y_batch = zip(*batch)
                    feed = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, step = sess.run([dis_train_op, dis_global_step], feed)

                except ValueError:
                    pass
    log.close()


if __name__ == "__main__":
    main()
