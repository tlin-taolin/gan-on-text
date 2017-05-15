# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse


def get_args():
    ROOT_DIRECTORY = '/home/tlin/notebooks/gan'
    ROOT_DIRECTORY = '/home/tlin/notebooks'
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, 'data')
    WORK_DIRECTORY = join(ROOT_DIRECTORY, 'code', 'demo5_v_textgan')
    DATA_DIRECTORY = join(WORK_DIRECTORY, 'data')
    TRAIN_DIR = join(DATA_DIRECTORY, 'training')

    """parameters for the model."""
    EMBEDDING_SIZE = 50
    OPTIMIZER_NAME = ['Adam', 'RMSProp'][0]

    """generate sentence for evaluation."""
    SAMPLING_TYPE = ['argmax', 'weighted_pick'][1]

    # feed them to the parser.
    parser = argparse.ArgumentParser()

    """define path."""
    parser.add_argument('--ROOT_DIRECTORY', type=str, default=ROOT_DIRECTORY)
    parser.add_argument('--RAW_DATA_DIRECTORY', type=str, default=RAW_DATA_DIRECTORY)
    parser.add_argument('--WORK_DIRECTORY', type=str, default=WORK_DIRECTORY)
    parser.add_argument('--DATA_DIRECTORY', type=str, default=DATA_DIRECTORY)
    parser.add_argument('--TRAIN_DIR', type=str, default=TRAIN_DIR)
    parser.add_argument('-i', '--INIT_FROM', type=str, default=None)
    parser.add_argument('--SVAE_TO', type=str, default=None)

    """tensorflow configuration."""
    parser.add_argument('--ALLOW_SOFT_PLACEMENT', action="store_false", default=True)
    parser.add_argument('--LOG_DEVICE_PLACEMENT', action="store_true", default=False)
    parser.add_argument('--GPU_MEM', type=float, default=0.666)

    """parameters for the training."""
    parser.add_argument('--SEED', type=int, default=666666)
    parser.add_argument('-d', '--DEBUG', action="store_true", default=False)
    parser.add_argument('--DEBUG_SIZE', type=int, default=150)
    parser.add_argument('-r', '--REBUILD_DATA', action="store_true", default=False)
    parser.add_argument('--SHUFFLE_DATA', action="store_true", default=False)
    parser.add_argument('--TRAIN_RATIO', type=float, default=0.8)

    """parameters for the model."""
    parser.add_argument('--RNN_LAYER', type=int, default=2)
    parser.add_argument('--RNN_SIZE', type=int, default=EMBEDDING_SIZE)
    parser.add_argument('--LABEL_DIM', type=int, default=1)
    parser.add_argument('--EMBEDDING_SIZE', type=int, default=EMBEDDING_SIZE)

    parser.add_argument('--EPOCH_PRETRAIN', type=int, default=5)
    parser.add_argument('--EPOCH_TRAIN', type=int, default=30)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--SENTENCE_LENGTH', type=int, default=25)
    parser.add_argument('-c', '--CLEAN_DATA', action="store_true", default=False)
    parser.add_argument('--MAX_VOCAB_SIZE', type=int, default=10000)
    # when < 1000, it will use default vocabulary without filtering.

    parser.add_argument('--OPTIMIZER_NAME', type=str, default=OPTIMIZER_NAME)
    parser.add_argument('--LEARNING_RATE_D', type=float, default=0.001)
    parser.add_argument('--LEARNING_RATE_G', type=float, default=0.001)
    parser.add_argument('--L2_REGULARIZATION_LAMBDA_D', type=float, default=1e-3)
    parser.add_argument('--L2_REGULARIZATION_LAMBDA_G', type=float, default=1e-3)
    parser.add_argument('--DROPOUT_RATE', type=float, default=0.5)
    parser.add_argument('--DECAY_RATE', type=float, default=0.97)
    parser.add_argument('--GRAD_CLIP', type=float, default=5.0)
    parser.add_argument('--SOFT_ARGMAX', type=str, default='100,10000')

    parser.add_argument('--D_ITERS_PER_BATCH', type=int, default=5)
    parser.add_argument('--G_ITERS_PER_BATCH', type=int, default=1)
    parser.add_argument('--WGAN_CLIP_VALUES', type=str, default='-0.01,0.01')
    parser.add_argument('--WGAN_GRADIENT_PENALTY', type=float, default=10.0)

    parser.add_argument('--D_CONV_SPATIALS', type=str, default='2,2,2,2')
    parser.add_argument('--D_CONV_DEPTHS', type=str, default='32,32,32,32')

    """parameters for evaluation."""
    parser.add_argument('--EVALUATE_EVERY', type=int, default=100)
    parser.add_argument('--CHECKPOINT_EVERY', type=int, default=5)
    parser.add_argument('--MAXNUM_MODEL_TO_KEEP', type=int, default=100)
    parser.add_argument('--BEAM_SEARCH', action="store_true", default=False)
    parser.add_argument('--BEAM_SEARCH_SIZE', type=int, default=5)
    parser.add_argument('--SAMPLING_TYPE', type=str, default=SAMPLING_TYPE)
    parser.add_argument('--SAMPLING_LENGTH', type=int, default=50)

    args = parser.parse_args()
    print_args(args)
    return args


def print_args(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))

if __name__ == '__main__':
    args = get_args()
    print(args.CLEAN_DATA)
