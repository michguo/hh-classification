# util.py

import tensorflow as tf

"""
Constants
"""
DATA_PATH = '/mnt0/data/aicare/lpch/npy/'
TRAIN_DATA_FILE = 'X_train.npy'
TRAIN_LABELS_FILE = 'y_train.npy'
VALIDATION_SIZE = 5000

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100
NUM_EPOCHS = 1#10

ACTION_COL = 5
NO_ACTION = 3
HAND_UNDER_DISP = 4
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

NUM_CHANNELS = 1
SEED = 66478
DTYPE = tf.float32
NUM_LABELS = 2

BASE_LR = 0.01


# Returns the shape of a np or tf tensor
def get_shape(x, name, type='tf'):
    if type == 'tf':
        print(name, ":", x.get_shape())
    else:
        print(name, ":", x.shape)


def fwrite(f, line):
    f.write(line)
    f.write('\n')
