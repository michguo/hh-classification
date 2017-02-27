import os
import sys
import time


import numpy as np
import tensorflow as tf


DATA_PATH = '/mnt0/data/aicare/lpch/npy/'
TRAIN_DATA_FILE = 'X_train.npy'
TRAIN_LABELS_FILE = 'y_train.npy'

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100
NUM_EPOCHS = 10

ACTION_COL = 5
NO_ACTION = 3
HAND_UNDER_DISP = 4
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

NUM_CHANNELS = 1
SEED = 66478
DTYPE = tf.float32
NUM_LABELS = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def getFilePath(filename):
    return DATA_PATH + filename


# Extract data from .npy files into numpy arrays
def extract_data(filepath):
    print('Extracting', filepath)
    data = np.load(filepath)
    print('Tensor dimensions:', data.shape)
    return data


# Filter out the examples that have some action other than NO_ACTION or HAND_UNDER_DISP
def filterExamples(inputs, labels):
    n_orig = inputs.shape[0]
    action_labels = labels[:, ACTION_COL]
    filter_condition = np.logical_or(action_labels == HAND_UNDER_DISP, action_labels == NO_ACTION)
    keep_idxs = np.where(filter_condition)
    inputs = inputs[keep_idxs]
    labels = labels[keep_idxs]
    print('Filtered out', n_orig - len(keep_idxs), 'examples')
    return (inputs, labels)


# Binarize action labels
def binarizedActionLabels(labels):
    return np.where(labels[:, ACTION_COL] == HAND_UNDER_DISP,
                    POSITIVE_LABEL,
                    NEGATIVE_LABEL)


def process_data(data, labels):
    data, labels = filterExamples(data, labels)
    labels = binarizedActionLabels(labels)
    data = np.expand_dims(data, axis=3)
    print('Finished processing data')
    print('Tensor dimensions are inputs:', data.shape, 'labels:', labels.shape)
    return (data, labels)


def train_val_split(data, labels):
    print('Creating train / val split')
    sections = [int(0.8 * len(data))]
    train_data, val_data = np.split(data, sections)
    train_labels, val_labels = np.split(labels, sections)
    return (train_data, train_labels, val_data, val_labels)


def get_shape(variable, name, type='tf'):
    if type == 'tf':
        print(name, ":", variable.get_shape())
    else:
        print(name, ":", variable.shape)


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  print("### Computing error ###")
  get_shape(predictions, 'predictions', type='np')
  get_shape(labels, 'labels', type='np')
  print(predictions)
  print(labels)
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):
    # Get the filepaths
    train_data_filepath = DATA_PATH + TRAIN_DATA_FILE
    train_labels_filepath = DATA_PATH + TRAIN_LABELS_FILE

    # Extract data from files
    data = extract_data(train_data_filepath)
    labels = extract_data(train_labels_filepath)

    # Process data
    # data, labels = process_data(data, labels)
    train_data, train_labels = process_data(data, labels)
    # train_data, train_labels, val_data, val_labels = train_val_split(data, labels)

    train_size = train_labels.shape[0]

    # Add placeholder variables
    print("### Creating placeholder variables ###")
    train_data_node = tf.placeholder(DTYPE,
                                     shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    get_shape(train_data_node, 'train_data_node')
    train_labels_node = tf.placeholder(tf.int64,
                                       shape=[BATCH_SIZE,])
    get_shape(train_labels_node, 'train_labels_node')
    eval_data = tf.placeholder(DTYPE,
                               shape=(EVAL_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
    get_shape(train_labels_node, 'train_labels_node')

    # Create parameter variables
    print("### Creating parameter variables ###")
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED, dtype=DTYPE))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=DTYPE))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=DTYPE))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=DTYPE))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_HEIGHT // 4 * IMAGE_WIDTH // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=DTYPE))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=DTYPE))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=DTYPE))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=DTYPE))


    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        print("### Conv Layer 1 ###")
        get_shape(data, 'data', type="np")
        get_shape(conv1_weights, 'conv1_weights')
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        get_shape(conv, 'conv1')
        # Bias and rectified linear non-linearity.
        print("### RELU Layer ###")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        print("### Pooling Layer ###")
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        print("### Conv Layer 1 ###")
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        print("### RELU Layer ###")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        print("### Pooling Layer ###")
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        print("### Fully Connected Layer 1 ###")
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            print("### Dropout ###")
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        print("### Fully Connected Layer 2 ###")
        return tf.matmul(hidden, fc2_weights) + fc2_biases
        # Training computation: logits + cross-entropy loss.

    logits = model(train_data_node, True)
    print('logits', logits)
    print('train_labels_node', train_labels_node)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=DTYPE)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)\
                                            .minimize(loss, global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)


    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print('Initialized!')
        # Loop through training steps.
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=main)