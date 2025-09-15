# Snakefile

import sys
sys.path.append("src")  # Add 'src' directory to the Python path
import os
import pickle
import numpy as np
import tensorflow as tf

# Load configuration from YAML
configfile: "config.yaml"

rule all:
    input:
        config["model_output"]

# Rule to load data
rule load_data:
    input:
        fasta_file=config["fasta_file"],
        go_annot_file=config["go_annot_file"]
    output:
        config["pdb2seq_file"],
        config["prot2annot_file"],
        config["goterms_file"],
        config["gonames_file"]
    run:
        from utils import load_data
        pdb2seq, prot2annot, goterms, gonames = load_data(input.fasta_file, input.go_annot_file)
        with open(output[0], 'wb') as f:
            pickle.dump(pdb2seq, f)
        with open(output[1], 'wb') as f:
            pickle.dump(prot2annot, f)
        with open(output[2], 'wb') as f:
            pickle.dump(goterms, f)
        with open(output[3], 'wb') as f:
            pickle.dump(gonames, f)

# Rule to load split data
rule load_split_data:
    input:
        train_file=config["train_file"],
        valid_file=config["valid_file"],
        test_file=config["test_file"]
    output:
        config["train_split_file"],
        config["valid_split_file"],
        config["test_split_file"]
    run:
        from utils import load_split_data
        train, valid, test = load_split_data(input.train_file, input.valid_file, input.test_file)
        with open(output[0], 'wb') as f:
            pickle.dump(train, f)
        with open(output[1], 'wb') as f:
            pickle.dump(valid, f)
        with open(output[2], 'wb') as f:
            pickle.dump(test, f)

# Rule to preprocess data
rule preprocess_data:
    input:
        config["pdb2seq_file"],
        config["prot2annot_file"],
        config["goterms_file"],
        config["gonames_file"],
        config["train_split_file"],
        config["valid_split_file"],
        config["test_split_file"]
    output:
        config["preprocessed_data_file"]
    run:
        from preprocess import preprocess, filter_and_label
        from utils import load_split_data, load_data

        pdb2seq, prot2annot, goterms, gonames = load_data(config["fasta_file"], config["go_annot_file"])
        train, valid, test = load_split_data(config["train_file"], config["valid_file"], config["test_file"])

        df, y_cut, df_grouped = preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test)
        df, y_cut = filter_and_label(df, prot2annot, goterms, gonames)

        with open(output[0], 'wb') as f:
            pickle.dump((df, y_cut, df_grouped), f)

# Rule to vectorize data
rule vectorize_data:
    input:
        config["preprocessed_data_file"]
    output:
        config["vectorized_data_file"]
    run:
        from vectorization import protein_vectorization
        with open(input[0], 'rb') as f:
            df, _, _ = pickle.load(f)

        sentences = df['seq']
        maxlen = config["output_sequence_length"]
        vocab_size = config["max_tokens"]

        vectorizer = protein_vectorization(sentences, max_tokens=vocab_size, output_sequence_length=maxlen)
        vectorized_words = vectorizer.vectorize_sequences()

        with open(output[0], 'wb') as f:
            pickle.dump(vectorized_words, f)

# Rule to prepare model data
rule prepare_model_data:
    input:
        preprocessed_data=config["preprocessed_data_file"],
        vectorized_data=config["vectorized_data_file"]
    output:
        dataset_train=config["dataset_train_dir"],
        dataset_val=config["dataset_val_dir"]
    params:
        batch_size=config["batch_size"]
    run:
        from prepare_model_data import prepare_model_data
        with open(input.preprocessed_data, 'rb') as f:
            df, y_cut, _ = pickle.load(f)
        with open(input.vectorized_data, 'rb') as f:
            vectorized_words = pickle.load(f)

        dataset_train, dataset_val = prepare_model_data(df, vectorized_words, y_cut, batch_size=params.batch_size)
        dataset_train.save(output.dataset_train)
        dataset_val.save(output.dataset_val)

# Rule to build and train the model
rule train_model:
    input:
        dataset_train=config["dataset_train_dir"],
        dataset_val=config["dataset_val_dir"]
    output:
        config["model_output"]
    params:
        epochs=config["epochs"],
        learning_rate=config["learning_rate"]
    run:
        from model_training import build_model, train_model

        train_dataset = tf.data.Dataset.load(input.dataset_train)
        val_dataset = tf.data.Dataset.load(input.dataset_val)

        model = build_model(train_dataset.element_spec[1],
                            maxlen=config["output_sequence_length"],
                            vocab_size=config["max_tokens"])

        train_model(model, train_dataset, val_dataset, epochs=params.epochs)
        model.save(output[0])
