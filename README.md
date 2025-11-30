# FEVER Baseline 2018 - Implementation

A clean implementation of the FEVER (Fact Extraction and VERification) baseline model from scratch, based on the original 2018 paper.

## Overview

This implementation recreates the FEVER baseline system which consists of:

1. **Document Retrieval**: TF-IDF-based retrieval (DrQA-style) to find k=5 most relevant Wikipedia pages
2. **Sentence Selection**: TF-IDF-based selection to find l=5 most relevant sentences from retrieved pages
3. **Textual Entailment**: Decomposable Attention model to classify claims as SUPPORTS, REFUTES, or NOT ENOUGH INFO

## Architecture

### Decomposable Attention Model

The model uses a three-step process (from Parikh et al. 2016):

- **Attend**: Soft alignment between claim and evidence using attention
- **Compare**: Element-wise comparison of aligned representations
- **Aggregate**: Sum and classify using feed-forward networks

**Hyperparameters** (from the original papers):
- Embedding dimension: 300 (GloVe)
- Projection dimension: 200
- Hidden dimension: 200
- Layers: 2
- Dropout: 0.2
- Optimizer: Adagrad
- Learning rate: 0.05

## Configuration

Edit [config.py](config.py) to control the system:

```python
# Set to True to download the full Wikipedia dump (50GB)
# Set to False to use oracle mode (assumes correct evidence is provided)
DOWNLOAD_WIKIPEDIA = False

# Retrieval parameters
NUM_DOCS_RETRIEVED = 5  # k in the paper
NUM_SENTENCES_RETRIEVED = 5  # l in the paper

# Model hyperparameters
EMBEDDING_DIM = 300
PROJECTION_DIM = 200
HIDDEN_DIM = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.05
NUM_EPOCHS = 10
```

## Installation

1. Create a Python virtual environment (Python 3.8+):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Everything at Once

```bash
python run.py all
```

This will:
1. Download the FEVER dataset
2. Train the model
3. Evaluate on dev set

### Option 2: Step-by-Step

**Step 1: Download Data**

```bash
python run.py download
```

This downloads:
- Training data (train.jsonl)
- Development data (dev.jsonl)
- Test data (test.jsonl)
- Wikipedia dump (wiki-pages.zip) - **only if `DOWNLOAD_WIKIPEDIA=True`**

**Step 2: Train the Model**

```bash
python run.py train
```

This will:
- Load the training data
- Retrieve evidence for each claim (using oracle mode or full retrieval)
- Build vocabulary
- Train the Decomposable Attention model
- Save the best model to `models/best_model.pt`

**Step 3: Evaluate the Model**

```bash
python run.py evaluate
```

Optional arguments:
```bash
python run.py evaluate --model models/best_model.pt --data data/dev.jsonl --output predictions.json
```

## Two Modes of Operation

### Mode 1: Oracle Mode (Default, DOWNLOAD_WIKIPEDIA=False)

- Does NOT download the 50GB Wikipedia dump
- Uses gold evidence from the dataset annotations
- Suitable for local development and testing
- Only tests the Decomposable Attention model performance

**Use this mode when:**
- You want to test the model locally
- You have limited storage/bandwidth
- You only care about the entailment component

### Mode 2: Full Retrieval Mode (DOWNLOAD_WIKIPEDIA=True)

- Downloads the full Wikipedia dump (50GB)
- Performs actual document retrieval and sentence selection
- Tests the complete FEVER pipeline
- Suitable for cloud computing with sufficient resources

**Use this mode when:**
- You want to test the full system
- You're running on a cloud instance with sufficient storage
- You want to evaluate retrieval performance

## Project Structure

```
fever_code/
├── config.py              # Configuration file
├── download_data.py       # Data download script
├── data_loader.py         # FEVER data loading utilities
├── wiki_processor.py      # Wikipedia page processing
├── retrieval.py           # Document retrieval & sentence selection
├── tokenizer.py           # Tokenization and vocabulary
├── model.py               # Decomposable Attention model
├── dataset.py             # PyTorch Dataset
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── run.py                 # Main entry point
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Data Format

The FEVER dataset uses JSONL format:

```json
{
  "id": 75397,
  "claim": "The Gadsden flag was named by Christopher Gadsden.",
  "label": "SUPPORTS",
  "evidence": [[[null, null, "Gadsden_flag", 0]]]
}
```

Labels:
- `SUPPORTS`: Evidence supports the claim
- `REFUTES`: Evidence refutes the claim
- `NOT ENOUGH INFO`: Insufficient evidence

## Expected Performance

Based on the original FEVER paper (2018):

- **With gold evidence (oracle)**: ~31.87% accuracy
- **Without gold evidence (full pipeline)**: ~50.91% accuracy

Note: These are baseline results. The model can be improved with:
- Pretrained embeddings (GloVe)
- Better hyperparameter tuning
- More training epochs
- Enhanced retrieval methods

## References

1. **FEVER Paper**: Thorne, James, et al. "FEVER: a large-scale dataset for Fact Extraction and VERification." NAACL-HLT (2018).
   - Paper: [https://aclanthology.org/N18-1074/](https://aclanthology.org/N18-1074/)
   - arXiv: [https://arxiv.org/abs/1803.05355](https://arxiv.org/abs/1803.05355)

2. **Decomposable Attention**: Parikh, Ankur P., et al. "A decomposable attention model for natural language inference." EMNLP (2016).
   - arXiv: [https://arxiv.org/abs/1606.01933](https://arxiv.org/abs/1606.01933)

3. **FEVER Dataset**: [https://fever.ai/dataset/fever.html](https://fever.ai/dataset/fever.html)

4. **Original Implementation**: [https://github.com/sheffieldnlp/naacl2018-fever](https://github.com/sheffieldnlp/naacl2018-fever)

## Cloud Deployment

For running the full system with Wikipedia retrieval, see the **comprehensive guide**: [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)

**Quick steps:**
1. Set `DOWNLOAD_WIKIPEDIA = True` in [config.py](config.py)
2. Upload code to cloud (AWS EC2, Google Cloud, Paperspace, etc.)
3. Run `bash setup_cloud.sh` on your cloud instance
4. Run `python run.py all` in a screen session

**Minimum Requirements:**
- 80GB storage (50GB Wikipedia + 30GB workspace)
- 16GB RAM minimum, 32GB recommended
- GPU highly recommended (T4, V100, or A100)
- Training time: ~3-5 hours with GPU vs ~20-35 hours on CPU

**Recommended Platforms:**
- Google Colab Pro (~$50/month)
- AWS EC2 p3.2xlarge (~$3/hour)
- Google Cloud with T4 GPU (~$0.60/hour)
- Paperspace Gradient (~$0.45/hour)

## License

This is an educational reimplementation of the FEVER baseline. Please cite the original papers if you use this code.

## Troubleshooting

**Issue**: "Wikipedia directory not found"
- Set `DOWNLOAD_WIKIPEDIA=True` in config.py or use oracle mode

**Issue**: "Vocabulary not found"
- Run training first: `python run.py train`

**Issue**: "CUDA out of memory"
- Reduce `BATCH_SIZE` in config.py
- Use CPU: set `DEVICE = "cpu"` in config.py

**Issue**: Training is slow
- Use GPU if available
- Reduce dataset size for testing
- In oracle mode, training should be faster
