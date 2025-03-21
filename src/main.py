import os
from utils import load_data, load_split_data
from preprocess import preprocess, filter_and_label
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
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Use base_dir to build the paths to data files
    fasta_file = os.path.join(base_dir, '..', 'data', 'pdb_seqres.txt.gz')
    go_annot_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_annot.tsv')
    train_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_train.txt')
    valid_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_valid.txt')
    test_file = os.path.join(base_dir, '..', 'data', 'nrPDB-GO_2019.06.18_test.txt')

    main(fasta_file, go_annot_file, train_file, valid_file, test_file)
