import pandas as pd
import numpy as np
import os
from DeepFRI.preprocessing.create_nrPDB_GO_annot import *
from DeepFRI.deepfrier.utils import *
from config.paths import TRAIN_SPLIT_FILE, VALID_SPLIT_FILE, TEST_SPLIT_FILE




def load_data():
    """
    Load and return the protein sequence data and GO annotations.
    """
    # Load pdb2seq from the file (make sure to adjust paths according to your structure)
    pdb2seq = read_fasta('Downloads/pdb_seqres.txt.gz')
    
    # Load GO annotations
    prot2annot, goterms, gonames, counts = load_GO_annot('DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv')
    goterms = goterms['mf']
    gonames = gonames['mf']
    
    return pdb2seq, prot2annot, goterms, gonames


def load_split_data(train_file, valid_file, test_file):
    """
    Load the train, validation, and test splits from specified files.

    Args:
    - train_file (str): Path to the train split file.
    - valid_file (str): Path to the validation split file.
    - test_file (str): Path to the test split file.

    Returns:
    - tuple: A tuple containing three lists (train, valid, test).
    """
    # Check if the files exist
    for file_path in [train_file, valid_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load the data and remove any empty lines
    def load_file(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    train = load_file(train_file)
    valid = load_file(valid_file)
    test = load_file(test_file)

    return train, valid, test

    
def preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test, maxlen=300):
    """
    Preprocess the protein sequence data and assign GO term labels.
    """
    df = pd.DataFrame.from_dict(pdb2seq, orient='index').reset_index()
    df.columns = ['name', 'seq']
    
    # Filter sequences by length
    df['length'] = df['seq'].map(lambda x: len(str(x)))
    df['seq'] = df['seq'].map(lambda x: ' '.join(list(x)))
    
    # Assign data types (train, valid, test)
    df['data_type'] = np.nan
    df['data_type'] = df['data_type'].astype('object')
    df.loc[df.name.isin(train), 'data_type'] = 'train'
    df.loc[df.name.isin(valid), 'data_type'] = 'valid'
    df.loc[df.name.isin(test), 'data_type'] = 'test'
    
    # Filter by sequence length and remove any NaNs
    df = df[df['length'] <= maxlen]
    df = df.dropna()
    
    # Assign labels based on GO term annotations
    y = np.array([prot2annot[n]['mf'] for n in df.name])
    df['label'] = [goterms[x] for x in y.argmax(axis=1)]
    df['idx'] = [x for x in y.argmax(axis=1)]
    
    # Filter data for single activation sequences
    index_keep = np.where(y.sum(axis=1) == 1)[0]
    df = df.iloc[index_keep]
    y_cut = y[index_keep]
    
    # Reset the index for the filtered dataframe
    df = df.reset_index(drop=True)
    
    # Group by label and add population size
    counts = df.groupby(['label'])['seq'].count().reset_index().rename(columns={'seq': 'population_size'})
    df = pd.merge(df, counts, on='label').reset_index()

    # Group by idx and label to get population sizes for each group
    df_grouped = df.groupby(["idx", "label"])[["name"]].count().reset_index()
    df_grouped = df_grouped.rename(columns={'name': 'population_size'}).sort_values("population_size")
    
    return df, y_cut, df_grouped

def filter_and_label(df, goterms, gonames, col_cut=[257,463,214,135]):
    """
    Filter and assign labels for specific GO-MF terms.
    """
    # Filter the data for specific GO-MF terms
    df = df[df.idx.isin(col_cut)].reset_index(drop=True)
    
    # Recalculate y_cut and assign new labels
    y_cut = np.array([prot2annot[n]['mf'] for n in df.name])
    y_cut = y_cut[:, col_cut]
    
    df['label'] = [goterms[x] for x in y_cut.argmax(axis=1)]
    df['goterm'] = [goterms[x] for x in y_cut.argmax(axis=1)]
    df['goname'] = [gonames[x] for x in y_cut.argmax(axis=1)]
    df['idx'] = [x for x in y_cut.argmax(axis=1)]
    
    return df, y_cut

def main():
    # Load all required data
    train_file = 'DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_train.txt'
    valid_file = 'DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_valid.txt'
    test_file = 'DeepFRI/preprocessing/data/nrPDB-GO_2019.06.18_test.txt'

    pdb2seq, prot2annot, goterms, gonames = load_data()
    train, valid, test = load_split_data(train_file, valid_file, test_file)

    # Preprocess the data
    df, y_cut, df_grouped = preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test)

    # Filter for specific GO-MF terms and reassign labels
    df, y_cut = filter_and_label(df, goterms, gonames)
    
    # Now df contains the final processed data
    print(df.head())
    print(df_grouped)

if __name__ == "__main__":
    main()
