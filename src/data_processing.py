import os
import csv
import gzip
import numpy as np
import pandas as pd
from Bio import SeqIO


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
        handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        if len(seq) >= 60 and len(seq) <= 1000:
            if len(set(seq).difference(aa)) == 0:
                prot2seq[prot] = seq

    return prot2seq

def load_GO_annot(filename):
    # Load GO annotations
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
                
    return prot2annot, goterms, gonames

def load_data(fasta_file, go_annot_file):
    """
    Load and return the protein sequence data and GO annotations.
    """
    pdb2seq = read_fasta(fasta_file)
    prot2annot, goterms, gonames = load_GO_annot(go_annot_file)
    goterms = goterms['mf']
    gonames = gonames['mf']

    return pdb2seq, prot2annot, goterms, gonames

def load_split_data(train_file, valid_file, test_file):
    """
    Load the train, validation, and test splits from specified files.
    """
    for file_path in [train_file, valid_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

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
    
    df['length'] = df['seq'].map(lambda x: len(str(x)))
    df['seq'] = df['seq'].map(lambda x: ' '.join(list(x)))
    
    df['data_type'] = np.nan
    df['data_type'] = df['data_type'].astype('object')
    df.loc[df.name.isin(train), 'data_type'] = 'train'
    df.loc[df.name.isin(valid), 'data_type'] = 'valid'
    df.loc[df.name.isin(test), 'data_type'] = 'test'
    
    df = df[df['length'] <= maxlen]
    df = df.dropna()
    
    y = np.array([prot2annot[n]['mf'] for n in df.name])
    df['label'] = [goterms[x] for x in y.argmax(axis=1)]
    df['idx'] = [x for x in y.argmax(axis=1)]
    
    index_keep = np.where(y.sum(axis=1) == 1)[0]
    df = df.iloc[index_keep]
    y_cut = y[index_keep]
    
    df = df.reset_index(drop=True)
    
    counts = df.groupby(['label'])['seq'].count().reset_index().rename(columns={'seq': 'population_size'})
    df = pd.merge(df, counts, on='label').reset_index()

    df_grouped = df.groupby(["idx", "label"])[["name"]].count().reset_index()
    df_grouped = df_grouped.rename(columns={'name': 'population_size'}).sort_values("population_size")
    
    return df, y_cut, df_grouped

def filter_and_label(df, prot2annot, goterms, gonames, col_cut=[257, 463, 214, 135]):
    """
    Filter and assign labels for specific GO-MF terms.
    """
    # Filter the dataframe based on idx values in col_cut
    df_filtered = df[df.idx.isin(col_cut)].reset_index(drop=True)
    
    # Update y_cut based on the filtered df's name
    y_cut = np.array([prot2annot[n]['mf'] for n in df_filtered.name])
    y_cut = y_cut[:, col_cut]
    
    # Assign the labels based on the filtered y_cut
    df_filtered['label'] = [goterms[x] for x in y_cut.argmax(axis=1)]
    df_filtered['goterm'] = [goterms[x] for x in y_cut.argmax(axis=1)]
    df_filtered['goname'] = [gonames[x] for x in y_cut.argmax(axis=1)]
    df_filtered['idx'] = [x for x in y_cut.argmax(axis=1)]
    
    return df_filtered, y_cut

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

if __name__ == "__main__":
    fasta_file = 'data/pdb_seqres.txt.gz'
    go_annot_file = 'data/nrPDB-GO_2019.06.18_annot.tsv'
    train_file = 'data/nrPDB-GO_2019.06.18_train.txt'
    valid_file = 'data/nrPDB-GO_2019.06.18_valid.txt'
    test_file = 'data/nrPDB-GO_2019.06.18_test.txt'

    main(fasta_file, go_annot_file, train_file, valid_file, test_file)
