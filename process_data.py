import numpy as np
import time

from settings import vdict
from util import *


###### DATA PREPROCESSING ######
# TODO: run this before training and save as new files?
def get_data():
    start_time = time.time()
    # Get the filepaths
    train_data_filepath = getFilePath(TRAIN_DATA_FILE)
    train_labels_filepath = getFilePath(TRAIN_LABELS_FILE)

    # Extract data from files
    data = extract_data(train_data_filepath)
    labels = extract_data(train_labels_filepath)

    # Process data
    train_data, train_labels = process_data(data, labels)

    # Create train/val split
    vdict['validation_data'] = train_data[:VALIDATION_SIZE, ...]
    vdict['validation_labels'] = train_labels[:VALIDATION_SIZE, ...]
    vdict['train_data'] = train_data[VALIDATION_SIZE:, ...]
    vdict['train_labels'] = train_labels[VALIDATION_SIZE:, ...]

    # vdict['train_data'], vdict['train_labels'], vdict['validation_data'], vdict['validation_labels'] = train_val_split(data, labels)

    vdict['train_size'] = vdict['train_labels'].shape[0]

    print('Took %.4f seconds' % (time.time() - start_time))


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
