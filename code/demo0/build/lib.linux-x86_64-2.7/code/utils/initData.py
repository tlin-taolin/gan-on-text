# -*- coding: utf-8 -*-
"""init data."""
import numpy as np


def init_x_real(para, mean_matrix, covariance_matrix):
    """init real data pdf."""
    return np.random.multivariate_normal(
        mean_matrix, covariance_matrix, para.NUM_SAMPLES)


def init_z(para):
    """init z sample."""
    return para.Z_PRIOR(
        -1, 1, [para.NUM_SAMPLES, para.Z_DIM]).astype(np.float32)


def batch_data(para, real_x_samples, z_samples, num_epochs=1, shuffle=True):
    """generate a batch iterator for a dataset."""
    num_batches_per_epoch = para.NUM_SAMPLES // para.BATCH_SIZE

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(para.NUM_SAMPLES))
        real_x_samples = real_x_samples[shuffle_indices]
        z_samples = z_samples[shuffle_indices]

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * para.BATCH_SIZE
        end_index = min((batch_num + 1) * para.BATCH_SIZE, para.NUM_SAMPLES)
        if start_index != end_index:
            yield real_x_samples[start_index:end_index], \
                z_samples[start_index:end_index]
