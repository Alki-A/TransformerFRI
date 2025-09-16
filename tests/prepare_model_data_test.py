import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from src.prepare_model_data import prepare_model_data  # replace with the actual import

def test_prepare_model_data():
    # Create dummy dataframe
    df = pd.DataFrame({
        "data_type": ["train", "train", "valid", "valid"],
        "sequence": ["ACD", "WYK", "GGG", "AAA"]
    })

    # Dummy vectorized sequences: shape (num_samples, sequence_length)
    vectorized_words = tf.constant([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=tf.int32)

    # Dummy labels
    y_cut = np.array([0, 1, 1, 0])

    batch_size = 2

    dataset_train, dataset_val = prepare_model_data(df, vectorized_words, y_cut, batch_size)

    # Check types
    assert isinstance(dataset_train, tf.data.Dataset)
    assert isinstance(dataset_val, tf.data.Dataset)

    # Collect batches
    X_train_list = []
    y_train_list = []
    for X_batch, y_batch in dataset_train:
        X_train_list.append(X_batch.numpy())
        y_train_list.append(y_batch.numpy())
    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)

    # Check shapes
    assert X_train_all.shape[0] == 2
    assert X_train_all.shape[1] == 3
    assert y_train_all.shape[0] == 2

    # Check values correspond to train indices
    np.testing.assert_array_equal(X_train_all, vectorized_words.numpy()[:2])
    np.testing.assert_array_equal(y_train_all, y_cut[:2])

    # Same checks for validation set
    X_val_list = []
    y_val_list = []
    for X_batch, y_batch in dataset_val:
        X_val_list.append(X_batch.numpy())
        y_val_list.append(y_batch.numpy())
    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val_all = np.concatenate(y_val_list, axis=0)

    # Check shapes
    assert X_val_all.shape[0] == 2
    assert X_val_all.shape[1] == 3
    assert y_val_all.shape[0] == 2

    # Check values correspond to validation indices
    np.testing.assert_array_equal(X_val_all, vectorized_words.numpy()[2:])
    np.testing.assert_array_equal(y_val_all, y_cut[2:])
