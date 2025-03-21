import pytest
import sys
import os
import numpy as np
import csv

# Add the 'src' directory to sys.path to allow imports from there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import read_fasta, load_GO_annot, load_data, load_split_data

# Test for the load_GO_annot function
def test_load_GO_annot():
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    go_annot_file_path = os.path.join(test_data_dir, 'go_annotations_mock_data.tsv')

    prot2annot, goterms, gonames = load_GO_annot(go_annot_file_path)

    # Ensure the loaded data contains expected keys
    assert len(prot2annot) > 0
    assert len(goterms) == 3  # 'mf', 'bp', 'cc'
    assert len(gonames) == 3   # 'mf', 'bp', 'cc'
    
    # Check that 'mf' (molecular function) contains expected GO terms
    assert "GO:0005126" in goterms['mf']  # This should match one of your test GO terms
    assert "GO:0008134" in goterms['mf']
    assert "GO:0016168" in goterms['mf']
    assert "GO:0004532" in goterms['mf']
    assert "GO:0003918" in goterms['mf']
    # Check that the protein annotations are correctly stored
    assert "TEST-1" in prot2annot
    assert len(prot2annot["TEST-1"]['mf']) > 0  # At least one molecular function annotation for TEST-1

# Test for the load_data function
def test_load_data():
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    fasta_file_path = os.path.join(test_data_dir, 'fasta_mock_data.fasta')
    go_annot_file_path = os.path.join(test_data_dir, 'go_annotations_mock_data.tsv')

    pdb2seq, prot2annot, goterms, gonames = load_data(fasta_file_path, go_annot_file_path)

    # Test that the loaded protein sequences are correct
    assert len(pdb2seq) == 3
    assert "TEST-1" in pdb2seq
    assert "TEST-2" in pdb2seq
    assert "TEST-3" in pdb2seq

    # Test that GO annotations are loaded correctly for each protein
    assert "TEST-1" in prot2annot
    assert "TEST-2" in prot2annot
    assert "TEST-3" in prot2annot

    # Test that the 'mf' annotations are available
    assert "mf" in prot2annot["TEST-1"]
    assert "mf" in prot2annot["TEST-2"]
    assert "mf" in prot2annot["TEST-3"]

    # Check if the 'mf' terms match expected GO terms
    assert np.any(prot2annot["TEST-1"]['mf'] == 1.0)
    assert np.any(prot2annot["TEST-2"]['mf'] == 1.0)
    assert np.any(prot2annot["TEST-3"]['mf'] == 1.0)

# Test for the load_split_data function
def test_load_split_data():
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    train_file_path = os.path.join(test_data_dir, 'train_mock_data.txt')
    valid_file_path = os.path.join(test_data_dir, 'valid_mock_data.txt')
    test_file_path = os.path.join(test_data_dir, 'test_mock_data.txt')

        # Create mock split files
    os.makedirs(test_data_dir, exist_ok=True)
    with open(train_file_path, 'w') as f:
        f.write("TEST-1\nTEST-2\nTEST-3\n")
    with open(valid_file_path, 'w') as f:
        f.write("TEST-4\nTEST-5\n")
    with open(test_file_path, 'w') as f:
        f.write("TEST-6\nTEST-7\n")


    # Test that the load_split_data function loads the splits correctly
    train, valid, test = load_split_data(train_file_path, valid_file_path, test_file_path)

    # Verify the content of the splits
    assert len(train) == 3
    assert len(valid) == 2
    assert len(test) == 2
    assert "TEST-1" in train
    assert "TEST-5" in valid
    assert "TEST-7" in test

    # Clean up mock files after test
    #os.remove(train_file_path)
    #os.remove(valid_file_path)
    #os.remove(test_file_path)
