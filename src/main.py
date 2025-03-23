import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import load_data, load_split_data
from preprocess import preprocess, filter_and_label
from vectorization import protein_vectorization
from prepare_model_data import prepare_model_data
from model_training import build_model, train_model

from config import (
    FASTA_FILE, GO_ANNOT_FILE, TRAIN_FILE, VALID_FILE, TEST_FILE,
    MAX_TOKENS, OUTPUT_SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_DROPOUT_RATE, MODEL_L2_REGULARIZATION
)
def main(fasta_file, go_annot_file, train_file, valid_file, test_file):
    """
    Main function to load, preprocess, and filter the data.
    """

    pdb2seq, prot2annot, goterms, gonames = load_data(fasta_file, go_annot_file)
    train, valid, test = load_split_data(train_file, valid_file, test_file)

    df, y_cut, df_grouped = preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test)

    df, y_cut = filter_and_label(df, prot2annot, goterms, gonames)
    
    print(df.head())
    print(df_grouped)
    

    # Create and adapt the vectorization layer
    #vectorize_layer = create_vectorization_layer(df.seq, MAX_TOKENS, OUTPUT_SEQUENCE_LENGTH)

    # Vectorize sequences
    #vectorized_sequences = vectorize_sequences(df['seq'].tolist(), vectorize_layer)
    #print(f"Shape of vectorized sequences: {vectorized_sequences.shape}")

    # Train the model using parameters from config
    sentences = df['seq']

    maxlen = 300  # Example sequence length (adjust based on your needs)
    vocab_size = len(set(''.join(sentences)))  # Number of unique amino acids in sequences

    # Create the vectorization object
    vectorizer = protein_vectorization(sentences, max_tokens=vocab_size, output_sequence_length=maxlen)
    
    # Vectorize the sequences
    vectorized_words = vectorizer.vectorize_sequences()
    print(f"Shape of vectorized_words: {len(vectorized_words)}")
    dataset_train, dataset_val = prepare_model_data(df, vectorized_words, y_cut, batch_size=BATCH_SIZE)
    print(f"Shape y_cut: {y_cut.shape}")

    model = build_model(y_cut, maxlen=OUTPUT_SEQUENCE_LENGTH, vocab_size=MAX_TOKENS)
    history = train_model(model, dataset_train, dataset_val, epochs=EPOCHS)

      # Expected: (batch_size, num_classes)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Use base_dir to build the paths to data files
    fasta_file = os.path.join(base_dir, '..', 'data', 'pdb_seqres.txt.gz')
    go_annot_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_annot.tsv')
    train_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_train.txt')
    valid_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_valid.txt')
    test_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_test.txt')

    main(fasta_file, go_annot_file, train_file, valid_file, test_file)
