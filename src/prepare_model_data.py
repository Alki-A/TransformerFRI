import tensorflow as tf

def prepare_model_data(df, vectorized_words, y_cut, batch_size, shuffle_buffer_size=90000):
    """
    Prepares the training and validation datasets.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        vectorized_words (tf.Tensor): The vectorized protein sequences.
        y_cut (np.ndarray): The labels for the sequences.

    Returns:
        X_train (np.ndarray): The training input data.
        X_val (np.ndarray): The validation input data.
        y_train (np.ndarray): The training labels.
        y_val (np.ndarray): The validation labels.
    """
    # Prepare training and validation data
    X_train = vectorized_words.numpy()[df[df.data_type == 'train'].index]
    X_val = vectorized_words.numpy()[df[df.data_type == 'valid'].index]
    
    # Labels
    y_train = y_cut[df[df.data_type == 'train'].index]
    y_val = y_cut[df[df.data_type == 'valid'].index]

    # Create TensorFlow datasets
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).shuffle(shuffle_buffer_size)


    return dataset_train, dataset_val
