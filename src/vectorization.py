import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.data import Dataset

class protein_vectorization:
    def __init__(self, sentences, max_tokens, output_sequence_length):
        """
        Initializes the ProteinVectorization class with the necessary configurations.

        Args:
            sentences (list): List of protein sequences.
            max_tokens (int): Maximum vocabulary size (number of unique amino acids).
            output_sequence_length (int): Maximum sequence length for padding/truncating.
        """
        self.sentences = sentences
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length

        # Create the TextVectorization layer
        self.vectorize_layer = TextVectorization(
            output_sequence_length=self.output_sequence_length,
            max_tokens=self.max_tokens
        )

        # Create a TensorFlow Dataset from sentences
        sentence_data = tf.convert_to_tensor(self.sentences, dtype=tf.string)
        
        # Adapt the vectorization layer to the sentences (learn the vocabulary)
        self.vectorize_layer.adapt(sentence_data)

    def vectorize_sequences(self):
        """
        Converts protein sequences into vectorized tokens.

        Returns:
            tf.Tensor: Vectorized sequences of protein sequences.
        """
        # Convert sentences into a tensor of strings
        word_tensors = tf.convert_to_tensor(self.sentences, dtype=tf.string)
        
        # Use the vectorization layer to vectorize the sequences
        vectorized_words = self.vectorize_layer(word_tensors)
        
        return vectorized_words

    def get_vocab(self):
        """
        Retrieves the vocabulary of the vectorizer.

        Returns:
            list: The vocabulary learned from the sentences.
        """
        return self.vectorize_layer.get_vocabulary()

