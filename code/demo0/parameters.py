# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join

import numpy as np

"""system part."""
WORK_DIRECTORY = "/home/tlin/notebooks/demo0"
TRAINING_DIRECTORY = join(WORK_DIRECTORY, "data")

"""tensorflow configuration."""
ALLOW_SOFT_PLACEMENT = True
LOG_DEVICE_PLACEMENT = False

"""parameters for reproduction."""
SEED = 666666

"""model parameters."""
Z_DIM = 100
Z_PRIOR = np.random.uniform
X_DIM = 2
D_ITERS_PER_G_ITERS = 5

"""train model."""
NUM_SAMPLES = 5000
BATCH_SIZE = 200
MAX_EPOCHS = 500
MIN_EPOCHS = 50
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
LEARNING_RATE_G_BETA = 0.5
LEARNING_RATE_D_BETA = 0.5
L2_REGULARIZATION_LAMBDA_G = 1e-3
L2_REGULARIZATION_LAMBDA_D = 1e-3
DROPOUT_RATE = 0.5
DECAY_RATE = 0.95
EARLY_STOPPING = 8

# check model.
EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 100
CHECKPOINT_DIRECTORY = TRAINING_DIRECTORY
