import os
import sys


from eval import *
from process_data import *
from model import *
from util import *


###### PLACEHOLDER VARIABLES ######
def add_placeholders():
    print("### Creating placeholder variables ###")

    vdict['train_data_node'] = tf.placeholder(DTYPE, shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    vdict['train_labels_node'] = tf.placeholder(tf.int64, shape=[BATCH_SIZE, ])
    vdict['eval_data_node'] = tf.placeholder(DTYPE, shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))


###### PARAMETER VARIABLES ######
def add_variables():
    print("### Creating parameter variables ###")

    vdict['conv1_weights'] = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                                             stddev=0.1,
                                                             seed=SEED,
                                                             dtype=DTYPE))
    vdict['conv1_biases'] = tf.Variable(tf.zeros([32], dtype=DTYPE))
    vdict['conv2_weights'] = tf.Variable(tf.truncated_normal([5, 5, 32, 64],
                                                             stddev=0.1,
                                                             seed=SEED,
                                                             dtype=DTYPE))
    vdict['conv2_biases'] = tf.Variable(tf.constant(0.1, shape=[64], dtype=DTYPE))
    vdict['fc1_weights'] = tf.Variable(tf.truncated_normal([IMAGE_HEIGHT // 4 * IMAGE_WIDTH // 4 * 64, 512],
                                                           stddev=0.1,
                                                           seed=SEED,
                                                           dtype=DTYPE))
    vdict['fc1_biases'] = tf.Variable(tf.constant(0.1, shape=[512], dtype=DTYPE))
    vdict['fc2_weights'] = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                           stddev=0.1,
                                                           seed=SEED,
                                                           dtype=DTYPE))
    vdict['fc2_biases'] = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=DTYPE))


def get_batch(step):
    offset = (step * BATCH_SIZE) % (vdict['train_size'] - BATCH_SIZE)
    batch_data = vdict['train_data'][offset:(offset + BATCH_SIZE), ...]
    batch_labels = vdict['train_labels'][offset:(offset + BATCH_SIZE)]
    return (batch_data, batch_labels)


def tf_compute_accuracy(yhat, labels):
    bools = tf.equal(yhat, labels)
    casted = tf.cast(bools, tf.float32)
    return tf.reduce_mean(casted)


def np_compute_accuracy(yhat, labels):
    bools = np.equal(yhat, labels)
    casted = bools.astype(np.float32)
    return np.mean(casted)


def main(_):
    get_data()
    train_size = vdict['train_size']

    # Building computation graph
    add_placeholders()
    add_variables()

    # Compute train accuracy #
    train_logits = model(vdict['train_data_node'], True)
    train_probs = tf.nn.softmax(train_logits)
    train_yhat = tf.argmax(train_probs, axis=1)
    train_accuracy = tf_compute_accuracy(train_yhat, vdict['train_labels_node'])

    # Compute eval accuracy #
    eval_logits = model(vdict['eval_data_node'])
    vdict['eval_probs'] = tf.nn.softmax(eval_logits)

    ###### OPTIMIZER ######
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=vdict['train_labels_node'],
                                                                         logits=train_logits))
    batch = tf.Variable(0, dtype=DTYPE)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(BASE_LR,  # Base learning rate.
                                               batch * BATCH_SIZE,  # Current index into the dataset.
                                               train_size,  # Decay step.
                                               0.95,  # Decay rate.
                                               staircase=True)
    # TODO regularization
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)


    # Create a local session to run the training.
    start_time = time.time()
    start_train = start_time
    outfile = open('./out/test.txt', 'w')
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        # Loop through training steps.
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            batch_data, batch_labels = get_batch(step)
            feed_dict = {vdict['train_data_node']: batch_data,
                         vdict['train_labels_node']: batch_labels}
            # Run the optimizer to update weights.
            _, l, lr, train_acc = sess.run([optimizer, loss, learning_rate, train_accuracy], feed_dict=feed_dict)

            epoch = float(step) * BATCH_SIZE / train_size
            elapsed_time = time.time() - start_time
            start_time = time.time()
            if step % EVAL_FREQUENCY == 0:
                # Get evaluation prdictions and compute accuracy
                eval_probs = eval_in_batches(vdict['validation_data'], sess)
                eval_yhat = np.argmax(eval_probs, axis=1)
                eval_acc = np_compute_accuracy(eval_yhat, vdict['validation_labels'])
                line = 'val_acc: %.4f, lr: %0.2e, loss: %0.4e, train_acc: %0.6f, (epoch %.2f), %.1f s' % (eval_acc, lr, l, train_acc, epoch, elapsed_time)
                print(line)
                fwrite(outfile, line)
            else:
                print('lr: %0.2e, loss: %0.4e, train_acc: %0.6f, (epoch %.2f), %.1f s' % (lr, l, train_acc, epoch, elapsed_time))
            sys.stdout.flush()
        line = "Training took a total of %.1f s" % (time.time() - start_train)
        print(line)
        fwrite(outfile, line)
        outfile.close()

if __name__ == '__main__':
    tf.app.run(main=main)

