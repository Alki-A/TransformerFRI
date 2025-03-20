import os

# Define the base path of the project (adjust this based on where the script is run from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your data directory (adjust as needed)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PREPROCESSING_DIR = os.path.join(DATA_DIR, 'preprocessing')

# Define paths for specific datasets or files
TRAIN_FILE = os.path.join(PREPROCESSING_DIR, 'nrPDB-GO_2019.06.18_train.txt')
VALID_FILE = os.path.join(PREPROCESSING_DIR, 'nrPDB-GO_2019.06.18_valid.txt')
TEST_FILE = os.path.join(PREPROCESSING_DIR, 'nrPDB-GO_2019.06.18_test.txt')
ANNOT_FILE = os.path.join(PREPROCESSING_DIR, 'nrPDB-GO_2019.06.18_annot.tsv')
PDB_SEQ_FILE = os.path.join(DATA_DIR, 'pdb_seqres.txt.gz')

# You can add more paths as needed (for model outputs, logs, etc.)
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'model_output')
