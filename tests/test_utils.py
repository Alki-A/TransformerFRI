import pytest
import sys
import os

# Add the 'src' directory to sys.path to allow imports from there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Bio import SeqIO
from src.utils import read_fasta

def test_read_fasta():
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    fasta_file_path = os.path.join(test_data_dir, 'fasta_mock_data.fasta')

    sequences = read_fasta(fasta_file_path)
    #sequences = {key.replace('-', '_'): value for key, value in sequences.items()}

    assert len(sequences) == 3
    assert "TEST-1" in sequences
    assert "TEST-2" in sequences
    assert "TEST-3" in sequences

