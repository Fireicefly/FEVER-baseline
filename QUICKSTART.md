# Quick Start Guide

This guide will help you get started with the FEVER baseline in 5 minutes.

## Setup (One-time)

1. **Verify your virtual environment**:
   ```bash
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the installation**:
   ```bash
   python test_setup.py
   ```

   You should see:
   ```
   [SUCCESS] All tests passed!
   ```

## Running the Baseline

### Option 1: Quick Test (Recommended for first run)

For local testing without downloading the 50GB Wikipedia dump:

```bash
python run.py all
```

This will:
- Download training/dev data (~100MB)
- Train the model in **oracle mode** (uses gold evidence)
- Evaluate on dev set
- Save the best model to `models/best_model.pt`

**Expected time**: 30-60 minutes (depending on your hardware)

### Option 2: Full Pipeline (Cloud recommended)

To test the complete system including document retrieval:

1. **Edit config.py**:
   ```python
   DOWNLOAD_WIKIPEDIA = True  # Change this line
   ```

2. **Run the full pipeline**:
   ```bash
   python run.py all
   ```

This will download 50GB of Wikipedia data and run the full retrieval pipeline.

**Expected time**: Several hours (depending on internet speed and hardware)

## Step-by-Step Mode

If you prefer to run each step separately:

```bash
# Step 1: Download data
python run.py download

# Step 2: Train model
python run.py train

# Step 3: Evaluate model
python run.py evaluate
```

## Configuration

Edit [config.py](config.py) to customize:

```python
# Toggle Wikipedia download (50GB)
DOWNLOAD_WIKIPEDIA = False  # Set to True for full pipeline

# Retrieval parameters
NUM_DOCS_RETRIEVED = 5      # Top-k documents
NUM_SENTENCES_RETRIEVED = 5  # Top-l sentences

# Model hyperparameters
BATCH_SIZE = 32             # Reduce if out of memory
NUM_EPOCHS = 10             # Increase for better performance
LEARNING_RATE = 0.05        # Adagrad learning rate

# Device
DEVICE = "cuda"             # Use "cpu" if no GPU
```

## Understanding the Two Modes

### Oracle Mode (DOWNLOAD_WIKIPEDIA=False)
- **Pros**: Fast, low storage requirements, good for testing
- **Cons**: Only tests the entailment model, not the retrieval
- **Use when**: Testing locally, limited resources

### Full Mode (DOWNLOAD_WIKIPEDIA=True)
- **Pros**: Tests the complete pipeline, realistic evaluation
- **Cons**: Requires 50GB+ storage, slow download
- **Use when**: Running on cloud, want to test retrieval

## What to Expect

### Oracle Mode Results
- Training time: 30-60 minutes (CPU), 10-20 minutes (GPU)
- Expected accuracy: ~30-35% (baseline performance)

### Full Pipeline Results
- Download time: 1-3 hours (depends on internet speed)
- Index building: 30-60 minutes
- Training time: Similar to oracle mode
- Expected accuracy: ~50-55% (baseline performance)

## Troubleshooting

**Problem**: "CUDA out of memory"
```python
# In config.py
BATCH_SIZE = 16  # or 8
DEVICE = "cpu"
```

**Problem**: "Data not found"
```bash
python run.py download
```

**Problem**: "Vocabulary not found"
```bash
python run.py train
```

**Problem**: Import errors
```bash
pip install -r requirements.txt
```

## File Structure After Running

```
fever_code/
├── data/
│   ├── train.jsonl           # Training data
│   ├── dev.jsonl             # Dev data
│   ├── test.jsonl            # Test data
│   └── wiki-pages/           # Wikipedia dump (if downloaded)
├── models/
│   ├── best_model.pt         # Best model checkpoint
│   └── vocab.pkl             # Vocabulary
└── [source files...]
```

## Next Steps

After running the baseline:

1. **Analyze results**: Check the confusion matrix to see which classes are confused
2. **Improve the model**: Try adding pretrained GloVe embeddings
3. **Tune hyperparameters**: Experiment with learning rate, batch size, epochs
4. **Enhance retrieval**: Implement better document retrieval methods

## Support

- Check [README.md](README.md) for detailed documentation
- Review test output: `python test_setup.py`
- Check configuration: `config.py`

## Key Files

- `config.py` - All configuration settings
- `run.py` - Main entry point
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `model.py` - Decomposable Attention model
- `retrieval.py` - Document retrieval & sentence selection
