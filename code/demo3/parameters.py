# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join

import numpy as np

"""system part."""
ROOT_DIRECTORY = '/home/tlin/notebooks'
# ROOT_DIRECTORY = '/home/tlin/notebooks/gan'
RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, 'data')
WORK_DIRECTORY = join(ROOT_DIRECTORY, 'code/demo3')
DATA_DIRECTORY = join(WORK_DIRECTORY, 'data')
TRAINING_DIRECTORY = join(DATA_DIRECTORY, 'training')

"""tensorflow configuration."""
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

"""parameters for reproduction."""
SEED = 666666
DEBUG = False
DEBUG_SIZE = 7200
REBUILD_DATA = False
SHUFFLE_DATA = True
RANDOM_SPLIT = True

"""model parameters."""
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1

LABEL_DIM = 1
EMBEDDING_SIZE = 30
RNN_SIZE = EMBEDDING_SIZE
RNN_DEPTH = 1
FORGET_GATE_BIASES = 1.0

Z_DIM = EMBEDDING_SIZE
Z_PRIOR = np.random.uniform
L_SOFT = 10000

D_ITERS_PER_BATCH = 1
G_ITERS_PER_BATCH = 5
WGAN_CLIP_VALUES = [-0.01, 0.01]

# convolutional part.
D_CONV_SPATIALS = [2, 2, 2, 2]
D_CONV_DEPTHS = [32, 32, 32, 32]

"""train model."""
NUM_SYMBOLS = 15
BATCH_SIZE = 256
IF_PRETRAIN = False
EPOCH_PRETRAIN = 50
EPOCH_TRAIN = 60
EPOCH_SENTENCE_GENERATION = 1

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
OPTIMIZER_NAME = ['Adam', 'RMSProp'][0]
L2_REGULARIZATION_LAMBDA_G = 1e-3
L2_REGULARIZATION_LAMBDA_D = 1e-3
DROPOUT_RATE = 0.5
DECAY_RATE = 0.95

# check model.
EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 5
CHECKPOINT_DIRECTORY = TRAINING_DIRECTORY

# sentence generation.
SENTENCE_LENGTH_TO_GENERATE = 10
