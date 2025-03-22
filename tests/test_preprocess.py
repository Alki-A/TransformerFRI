import pytest
import os
import numpy as np
import pandas as pd

# Add the 'src' directory to sys.path to allow imports from there
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess
from src.utils import load_GO_annot, load_data, load_split_data

# Test the preprocess function
def test_preprocess():
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    fasta_file_path = os.path.join(test_data_dir, 'fasta_mock_data.fasta')
    go_annot_file_path = os.path.join(test_data_dir, 'go_annotations_mock_data.tsv')
    
    # Load the data using the existing utils functions
    pdb2seq, prot2annot, goterms, gonames = load_data(fasta_file_path, go_annot_file_path)
    
    # Define train, valid, and test splits (using existing mock splits from utils test)
    train = ['TEST-1', 'TEST-2']
    valid = ['TEST-3']
    test = []
    
    # Run preprocess with the loaded data
    df, y_cut, df_grouped = preprocess(pdb2seq, prot2annot, goterms, gonames, train, valid, test)
    
    # Check the resulting dataframe
    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert 'name' in df.columns, "DataFrame should have a 'name' column"
    assert 'seq' in df.columns, "DataFrame should have a 'seq' column"
    assert 'label' in df.columns, "DataFrame should have a 'label' column"
    
    # Check the length of the output data
    assert len(df) == 1, "One entry meets all our criteria"
    
    # Check if the y_cut is the expected shape
    assert y_cut.shape == (1, 5), "y_cut should have 1 records and 5 classes"

    # Check the grouped dataframe
    assert isinstance(df_grouped, pd.DataFrame), "df_grouped should be a pandas DataFrame"
    assert 'population_size' in df_grouped.columns, "df_grouped should have 'population_size' column"
    
    # Check if label assignment is correct
    assert df['label'].iloc[0] == 'GO:0004532', "First and only record should have label 'GO:0004532'"
 
