"""
model_training.py

Handles model training for protein sequence classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformer import TransformerBlock, TokenAndPositionEmbedding

def build_model(y, maxlen, vocab_size, embed_dim=1024, num_heads=12, ff_dim=1024):
    """
    Builds and compiles the Transformer-based model.
    
    Args:
        maxlen (int): Maximum sequence length.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        ff_dim (int): Hidden layer size in the feed-forward network.
    
    Returns:
        keras.Model: Compiled model.
    """
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.0)
    x = transformer_block(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(y.shape[1], activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, dataset_train, dataset_val, epochs=200):
    """
    Trains the model.
    
    Args:
        model (keras.Model): Compiled model.
        dataset_train (tf.data.Dataset): Training dataset.
        dataset_val (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs.
    
    Returns:
        History: Training history.
    """
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
    )
    return history
