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


    train_indices = df[df.data_type == 'train'].index
    val_indices = df[df.data_type == 'valid'].index

    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of validation samples: {len(val_indices)}")


    # Prepare training and validation data
    X_train = vectorized_words.numpy()[df[df.data_type == 'train'].index]
    X_val = vectorized_words.numpy()[df[df.data_type == 'valid'].index]
    # Check the shapes of the sliced training and validation data

    
    # Labels
    y_train = y_cut[df[df.data_type == 'train'].index]
    y_val = y_cut[df[df.data_type == 'valid'].index]


    # Create TensorFlow datasets
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).shuffle(shuffle_buffer_size)

    print(f"Train samples: {df[df.data_type == 'train'].shape[0]}")
    print(f"Validation samples: {df[df.data_type == 'valid'].shape[0]}")



    return dataset_train, dataset_val
