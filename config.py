"""
Configuration for FEVER Baseline 2018

This config controls data download and model parameters.

DOWNLOAD_WIKIPEDIA modes:
  - False (Oracle mode): Uses gold evidence from dataset annotations
    * Faster, works on local machines
    * Good for testing and development
    * Skips Wikipedia download and retrieval

  - True (Full pipeline): Downloads Wikipedia and performs actual retrieval
    * Requires 60GB+ disk space
    * Recommended for cloud deployment
    * Computes real FEVER scores with retrieval
    * Expected to take 3-5 hours with GPU

For cloud deployment: Set DOWNLOAD_WIKIPEDIA=True
"""

# Data Download Settings
DOWNLOAD_WIKIPEDIA = True  # CHANGE TO TRUE FOR CLOUD DEPLOYMENT

# Data URLs
TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
TEST_URL = "https://fever.ai/download/fever/shared_task_test.jsonl"
WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"

# Data Paths
DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/train.jsonl"
DEV_FILE = f"{DATA_DIR}/dev.jsonl"
TEST_FILE = f"{DATA_DIR}/test.jsonl"
WIKI_DIR = f"{DATA_DIR}/wiki-pages"

# Model Paths
MODEL_DIR = "models"
GLOVE_PATH = f"{DATA_DIR}/glove.840B.300d.txt"  # Download separately if needed

# Retrieval Parameters (from FEVER paper)
NUM_DOCS_RETRIEVED = 5  # k in the paper
NUM_SENTENCES_RETRIEVED = 5  # l in the paper

# Decomposable Attention Hyperparameters (from Parikh et al. 2016)
EMBEDDING_DIM = 300  # GloVe embedding dimension
PROJECTION_DIM = 200  # Project embeddings down to 200 dimensions
HIDDEN_DIM = 200  # Hidden layer dimension
NUM_LAYERS = 2  # Number of layers in feed-forward networks
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.05
NUM_EPOCHS = 10
NUM_CLASSES = 3  # SUPPORTS, REFUTES, NOT ENOUGH INFO

# Device
DEVICE = "cuda" # Will fall back to CPU if CUDA not available
