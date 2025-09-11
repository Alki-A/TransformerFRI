# Snakefile

import sys
sys.path.append("src")  # Add 'src' directory to the Python path
import os
import pickle
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 3 to suppress INFO, WARNING, and ERROR logs
import tensorflow as tf
# Hardcoded configuration parameters
FASTA_FILE = 'data/pdb_seqres.txt.gz'
GO_ANNOT_FILE = 'data/nrPDB-GO_2019.06.18_annot.tsv'
TRAIN_FILE = 'data/nrPDB-GO_2019.06.18_train.txt'
VALID_FILE = 'data/nrPDB-GO_2019.06.18_valid.txt'
TEST_FILE = 'data/nrPDB-GO_2019.06.18_test.txt'

MAX_TOKENS = 5000
OUTPUT_SEQUENCE_LENGTH = 300
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.001
MODEL_DROPOUT_RATE = 0.5
MODEL_L2_REGULARIZATION = 0.01

rule all:
    input:
        "data/model.h5"
# Define rule to load data
rule load_data:
    input:
        fasta_file=FASTA_FILE,
        go_annot_file=GO_ANNOT_FILE
    output:
        "data/pdb2seq.pkl",
        "data/prot2annot.pkl",
        "data/goterms.pkl",
        "data/gonames.pkl"
    run:
        print("hello2")
        from utils import load_data
        pdb2seq, prot2annot, goterms, gonames = load_data(input.fasta_file, input.go_annot_file)
        print(f"Loaded data:")
        print(f"pdb2seq: {len(pdb2seq)} entries")
        print(f"prot2annot: {len(prot2annot)} entries")
        print(f"goterms: {len(goterms)} terms")
        print(f"gonames: {len(gonames)} names")
        with open(output[0], 'wb') as f:
            pickle.dump(pdb2seq, f)
        with open(output[1], 'wb') as f:
            pickle.dump(prot2annot, f)
        with open(output[2], 'wb') as f:
            pickle.dump(goterms, f)
        with open(output[3], 'wb') as f:
            pickle.dump(gonames, f)
        print("hello2")

# Define rule to load split data
rule load_split_data:
    input:
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        test_file=TEST_FILE
    output:
        "data/train.pkl",
        "data/valid.pkl",
        "data/test.pkl"
    run:
        from utils import load_split_data
        train, valid, test = load_split_data(input.train_file, input.valid_file, input.test_file)
        print(f"Loaded split data:")
        print(f"train: {len(train)} samples")
        print(f"valid: {len(valid)} samples")
        print(f"test: {len(test)} samples")

        with open(output[0], 'wb') as f:
            pickle.dump(train, f)
        with open(output[1], 'wb') as f:
            pickle.dump(valid, f)
        with open(output[2], 'wb') as f:
            pickle.dump(test, f)
        print("Data loading completed")

# Define rule to preprocess data
rule preprocess_data:
    input:
        "data/pdb2seq.pkl",
        "data/prot2annot.pkl",
        "data/goterms.pkl",
        "data/gonames.pkl",
        "data/train.pkl",
        "data/valid.pkl",
        "data/test.pkl"
    output:
        "data/preprocessed_data.pkl"
    run:
        from preprocess import preprocess, filter_and_label
        from utils import load_split_data
        from utils import load_data
        pdb2seq, prot2annot, goterms, gonames = load_data(FASTA_FILE, GO_ANNOT_FILE)
        train, valid, test = load_split_data(TRAIN_FILE, VALID_FILE, TEST_FILE)
        df, y_cut, df_grouped = preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test)
        df, y_cut = filter_and_label(df, prot2annot, goterms, gonames)
        with open(output[0], 'wb') as f:
            pickle.dump((df, y_cut, df_grouped), f)

# Define rule to vectorize data
rule vectorize_data:
    input:
        "data/preprocessed_data.pkl"
    output:
        "data/vectorized_data.pkl"
    run:
        from vectorization import protein_vectorization
        with open(input[0], 'rb') as f:
            df, _, _ = pickle.load(f)
        
        sentences = df['seq']
        maxlen = 300  # Adjust sequence length
        vocab_size = len(set(''.join(sentences)))
        
        # Vectorize sequences
        vectorizer = protein_vectorization(sentences, max_tokens=vocab_size, output_sequence_length=maxlen)
        vectorized_words = vectorizer.vectorize_sequences()
        print("Data type:", type(vectorized_words))
        print(f"Shape of vectorized_words: {np.array(vectorized_words).shape}")

        print("helo")
        with open(output[0], 'wb') as f:
            pickle.dump(vectorized_words, f)

# Define rule to prepare model data
rule prepare_model_data:
    input:
        preprocessed_data="data/preprocessed_data.pkl",
        vectorized_data="data/vectorized_data.pkl"
    output:
        dataset_train=directory("data/dataset_train"),
        dataset_val=directory("data/dataset_val")
    params:
        batch_size=BATCH_SIZE
    run:
        from prepare_model_data import prepare_model_data
        import pickle
        import tensorflow as tf

        # Load preprocessed data and vectorized words
        with open(input.preprocessed_data, 'rb') as f:
            df, y_cut, _ = pickle.load(f)

        with open(input.vectorized_data, 'rb') as f:
            vectorized_words = pickle.load(f)

        # Prepare training and validation datasets
        dataset_train, dataset_val = prepare_model_data(df, vectorized_words, y_cut, batch_size=params.batch_size)

        # Save datasets using tf.data.Dataset.save
        dataset_train.save(output.dataset_train)
        dataset_val.save(output.dataset_val)


# Define rule to build and train the model
rule train_model:
    input:
        dataset_train="data/dataset_train",
        dataset_val="data/dataset_val"
    output:
        "data/model.h5"
    params:
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    run:
        from model_training import build_model, train_model
        import tensorflow as tf

        # Load datasets
        train_dataset = tf.data.Dataset.load(input.dataset_train)
        val_dataset = tf.data.Dataset.load(input.dataset_val)

        # Build the model
        model = build_model(train_dataset.element_spec[1], maxlen=OUTPUT_SEQUENCE_LENGTH, vocab_size=MAX_TOKENS)

        # Train the model using the loaded datasets
        history = train_model(model, train_dataset, val_dataset, epochs=params.epochs)

        # Save the trained model
        model.save(output[0])

