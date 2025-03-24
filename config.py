# config.py

# Paths to data files
FASTA_FILE = '../data/pdb_seqres.txt.gz'
GO_ANNOT_FILE = '../data/nrPDB-GO_2019.06.18_annot.tsv'
TRAIN_FILE = '../data/nrPDB-GO_2019.06.18_train.txt'
VALID_FILE = '../data/nrPDB-GO_2019.06.18_valid.txt'
TEST_FILE = '../data/nrPDB-GO_2019.06.18_test.txt'

# Vectorization settings
MAX_TOKENS = 10000  # Maximum vocabulary size for the TextVectorization layer
OUTPUT_SEQUENCE_LENGTH = 300  # Maximum sequence length for padding/truncation

# Training settings
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 1  # Number of epochs for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# Model settings (adjust these as needed for your model architecture)
MODEL_DROPOUT_RATE = 0.2  # Dropout rate for regularization
MODEL_L2_REGULARIZATION = 1e-5  # L2 regularization strength

# File paths for saving trained models or logs
#MODEL_SAVE_PATH = '../models/trained_model.h5'
#LOG_DIR = '../logs/'
