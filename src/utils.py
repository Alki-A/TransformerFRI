import os
import csv
import gzip
import numpy as np
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
    def load_file(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    train = load_file(train_file)
    valid = load_file(valid_file)
    test = load_file(test_file)

    return train, valid, test
