import numpy as np
import cPickle

from re import compile as _Re

class Dis_dataloader(object):
    def __init__(self, vocab_size=5000, sequence_len=20):
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len

    def load_data_n_labels(self, positive_file, negative_file):
        ''' positive from the real data, negative is generated from G '''
        positive_egs = []
        negative_egs = []
        with open(positive_file) as p:
            for line in p:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                positive_egs.append(parse_line)
        p.close()

        with open(negative_file) as n:
            for line in n:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.sequence_len:
                    negative_egs.append(parse_line)
        n.close()

        positive_labs = [[0, 1] for _ in positive_egs]
        negative_labs = [[1, 0] for _ in negative_egs]

        y = np.array(np.concatenate([positive_labs, negative_labs], 0))
        x_test = np.array(positive_egs + negative_egs)

        return[x_test, y]

    def load_train_data(self, positive_file, negative_file):
        ''' The training data for D consists of the real data and the data
            generated from the generator. Shuffle the training data. '''
        sentences, labels = self.load_data_n_labels(positive_file, negative_file)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = sentences[shuffle_indices]
        y_shuffled = labels[shuffle_indices]
        return [x_shuffled, y_shuffled]

    def load_test_data(self, positive_file, test_file):
        test_egs = []
        test_labs = []
        with open(test_file) as te:
            for line in te:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                test_egs.append(parse_line)
                test_labs.append([1, 0])
        te.close()
        with open(positive_file) as p:
            for line in p:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                test_egs.append(parse_line)
                test_labs.append([0, 1])
        p.close()
        test_egs = np.array(test_egs)
        test_labs = np.array(test_labs)
        shuffle_indices = np.random.permutation(np.arange(len(test_labs)))
        x_dev = test_egs[shuffle_indices]
        y_dev = test_labs[shuffle_indices]
        return [x_dev, y_dev]

    def batch_iteror(self, data, batch_size, num_epochs):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in xrange(num_epochs):
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            for batch_idx in xrange(num_batches_per_epoch):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, data_size)
                yield shuffled_data[start_idx:end_idx]
