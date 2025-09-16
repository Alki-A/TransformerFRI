import tensorflow as tf
import pytest
from src.vectorization import protein_vectorization
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_vectorization_shapes_and_vocab():
    sequences = ["ACD", "WYK", "GGG"]

    max_tokens = 10
    maxlen = 5

    vec = protein_vectorization(sequences, max_tokens=max_tokens, output_sequence_length=maxlen)
    vectorized = vec.vectorize_sequences()

    assert vectorized.shape == (len(sequences), maxlen)
    assert vectorized.dtype in [tf.int32, tf.int64]

    vocab = vec.get_vocab()
    # Check that at least one of the original sequences appears in vocab
    assert any(seq.lower() in vocab for seq in ["acd", "wyk", "ggg"])
    assert "[UNK]" in vocab

def test_vectorization_consistency():
    sequences = ["AAA", "AAA"]
    vec = protein_vectorization(sequences, max_tokens=5, output_sequence_length=3)

    vectorized = vec.vectorize_sequences().numpy()

    # Both sequences are identical, so their encodings should match
    assert (vectorized[0] == vectorized[1]).all()
