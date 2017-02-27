import numpy as np
import tensorflow as tf

DATA_PATH = '/mnt0/data/aicare/lpch/npy/'
TRAIN_DATA_FILE = 'X_train.npy'
TRAIN_LABELS_FILE = 'y_train.npy'

BATCH_SIZE = 64

ACTION_COL = 5
NO_ACTION = 3
HAND_UNDER_DISP = 4
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

def getFilePath(filename):
    return DATA_PATH + filename

# Extract data from .npy files into numpy arrays
def extract_data(filepath):
    print('Extracting', filepath)
    data = np.load(filepath)
    print ('Tensor dimensions:', data.shape)
    return data

# Filter out the examples that have some action other than NO_ACTION or HAND_UNDER_DISP
def filterExamples(inputs, labels):
    n_orig = inputs.shape[0]
    action_labels = labels[:,ACTION_COL]
    filter_condition = np.logical_or(action_labels == HAND_UNDER_DISP, action_labels == NO_ACTION)
    keep_idxs = np.where(filter_condition)
    inputs = inputs[keep_idxs]
    labels = labels[keep_idxs]
    print('Filtered out', n_orig-len(keep_idxs), 'examples') 
    return (inputs, labels)

# Binarize action labels
def binarizedActionLabels(labels):
    return np.where(labels[:,ACTION_COL] == HAND_UNDER_DISP,
                                     POSITIVE_LABEL,
                                     NEGATIVE_LABEL)

def process_data(inputs, labels):
    inputs, labels = filterExamples(inputs, labels)
    labels = binarizedActionLabels(labels)
    print('Finished processing data')
    print('Tensor dimensions are inputs:', inputs.shape, 'labels:', labels.shape)
    return (inputs, labels)

def main(_):
    # Get the filepaths
    train_data_filepath = DATA_PATH + TRAIN_DATA_FILE
    train_labels_filepath = DATA_PATH + TRAIN_LABELS_FILE 
    
    # Extract data from files
    train_data = extract_data(train_data_filepath)
    train_labels = extract_data(train_labels_filepath)

    # Process data
    train_data, train_labels = process_data(train_data, train_labels)

    # Add placeholder variables
    _, height, width = train_data.shape
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, height, width))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

if __name__ == '__main__':
   tf.app.run(main=main) 
