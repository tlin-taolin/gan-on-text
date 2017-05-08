import tensorflow as tf
import numpy as np


def generate_samples(sess, trainable, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in xrange(int(generated_num / batch_size)):
        generated_samples.extend(trainable.generate(sess))
    with open(output_file, "w") as fout:
        for item in generated_samples:
            temp = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(temp)
    fout.close()


def pre_train_epoch(sess, trainable, data_loader):
    supervised_g_loss = []
    data_loader.reset_pointer()

    for batch in xrange(data_loader.num_batch):
        if batch % 100 == 0:
            print "%d / %d" % (batch, data_loader.num_batch)
            print "Training loss : ", np.mean(supervised_g_loss)
        next_bc = data_loader.next_batch()
        ''' [pretrain_update, pretrain_loss] '''
        _, g_loss = trainable.pretrain_step(sess, next_bc)
        supervised_g_loss.append(g_loss)

    return np.mean(supervised_g_loss)
