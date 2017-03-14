from gen_dataloader import Gen_Data_loader, Likelihood_Data_loader
from target_lstm import TARGET_LSTM
# from overrides import overrides

import model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import random

# G parameters
EBD_DIM = 32
HID_DIM = 32
SEQ_LEN = 20
START_TOKEN = 0

PRE_EPC_NUM = 350
TRAIN_ITER = 1
SEED = 88
BATCH_SIZE = 64

# paths of real_data, G_output, evaluating file
positive_file = "save/real_data.txt"
negative_file = "target_generate/generator_sample.txt"
eval_file = "target_generate/eval_file" # output file of our G

generated_num = 10000

class G_(model.LSTM):
    # @overrides
    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()

def get_trainable_model(num_emb):
    return G_(num_emb, BATCH_SIZE, EBD_DIM, HID_DIM, SEQ_LEN, START_TOKEN)

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    ''' outputs from G '''
    generated_samples = []
    for _ in xrange(int(generated_num/batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    with open(output_file, "w") as f:
        # print generated_samples
        for wd in generated_samples:
            connected = " ".join([str(x) for x in wd]) + "\n"
            f.write(connected)
    f.close()

def target_loss(sess, target_lstm, data_loader):
    ''' obtain the loss from the target_lstm, this would be the target '''
    supervised_g_losses = []
    data_loader.reset_pointer()

    for _ in xrange(data_loader.num_batch):
        Batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: Batch})
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def pre_train_epoch(sess, trainable_model, data_loader):
    ''' obtain the loss from G '''
    supervised_g_losses = []
    data_loader.reset_pointer()

    for _ in xrange(data_loader.num_batch):
        Batch = data_loader.next_batch()
        _, g_loss, g_pred = trainable_model.pretrain_step(sess, Batch)
        supervised_g_losses.append(g_loss)

    print " ===>> G train loss: ", np.mean(supervised_g_losses)
    return np.mean(supervised_g_losses)

def pre_training():
    random.seed(SEED)
    np.random.seed(SEED)

    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LEN)
    lik_data_loader = Likelihood_Data_loader(BATCH_SIZE, SEQ_LEN)
    vocab_size = 5000

    G = get_trainable_model(vocab_size)
    target_params = cPickle.load(open("save/target_params.pkl"))
    target_lstm = TARGET_LSTM(vocab_size, 64, 32, 32, 20, 0, target_params)
    # num_emb, batch_size, emb_dim, hidden_dim, sequence_len, start_token, params

    # experiment config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # using the target_lstm to generate real data
    generate_samples(sess, target_lstm, 64, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    print "Finish preparation for pretraining..."

    log = open("log/experiment-log.txt", "w")
    print "Start pretrain the generator ... "
    log.write("Pre-training...\n")

    for epoch in xrange(PRE_EPC_NUM):
        print "Pretrain epoch: ", epoch
        loss = pre_train_epoch(sess, G, gen_data_loader)
        if epoch % 2 == 0:
            generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file)
            lik_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, lik_data_loader)
            print "pre-train epoch: ", epoch, "test_loss: ", test_loss
            connexion = str(epoch) + " " + str(test_loss) + "\n"
            log.write(connexion)

    print "test the obtained model ... "
    generate_samples(sess, G, BATCH_SIZE, generated_num, eval_file) # generate from G
    lik_data_loader.create_batches(eval_file)
    test_loss = target_loss(sess, target_lstm, lik_data_loader)
    connexion = "After supervised trian, loss is : " + " " + str(test_loss) + "\n"
    log.write(connexion)
    log.close()


if __name__ == "__main__":
    pre_training()
