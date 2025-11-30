# FEVER Baseline - Cloud Deployment Guide

This guide explains how to deploy the FEVER baseline on a cloud computing server with full Wikipedia download and faster computing.

## Resource Requirements

**Minimum specifications:**
- **Storage**: 80GB (50GB Wikipedia + 20GB model/data/temp)
- **RAM**: 16GB minimum, 32GB recommended
- **CPU**: 8+ cores recommended
- **GPU**: Highly recommended (training 10x faster)
  - NVIDIA GPU with 8GB+ VRAM (e.g., T4, V100, A100)
  - CUDA 11.0+ support

**Estimated costs (per hour):**
- Google Cloud (n1-highmem-8 + T4 GPU): ~$0.60/hour
- AWS EC2 (p3.2xlarge): ~$3.06/hour
- Azure (NC6s_v3): ~$0.90/hour
- Paperspace Gradient: ~$0.45/hour

## Recommended Cloud Platforms

### Option 1: Google Colab Pro+ (Easiest, Good for Testing)
**Pros**: Easy setup, Jupyter interface, no server management
**Cons**: Session timeouts, limited to ~24 hours, shared resources
**Cost**: $50/month subscription

### Option 2: AWS EC2 (Most Flexible)
**Pros**: Full control, many instance types, pay-per-use
**Cons**: Requires more setup, AWS knowledge helpful
**Cost**: Pay per hour (~$0.60-$3/hour depending on instance)

### Option 3: Google Cloud Compute Engine
**Pros**: Good GPU availability, free $300 credit for new users
**Cons**: Requires Google Cloud account setup
**Cost**: Similar to AWS

### Option 4: Paperspace Gradient (Recommended for ML)
**Pros**: ML-focused, easy setup, good documentation
**Cons**: Smaller provider, limited regions
**Cost**: ~$0.45/hour for GPU instances

## Quick Start - Deployment Steps

### Step 1: Prepare Your Codebase

1. **Create a git repository** (recommended):
   ```bash
   cd c:\Users\Younes\Desktop\fever_code
   git init
   git add .
   git commit -m "Initial FEVER baseline implementation"
   ```

   Then push to GitHub/GitLab:
   ```bash
   git remote add origin https://github.com/yourusername/fever-baseline.git
   git push -u origin main
   ```

2. **OR zip your code** (simpler but less flexible):
   - Zip the entire `fever_code` folder (exclude `.venv`, `data/`, `models/`)
   - Upload to Google Drive or transfer via scp

### Step 2: Enable Wikipedia Download

Edit `config.py` and change:
```python
DOWNLOAD_WIKIPEDIA = True  # Changed from False to True
```

### Step 3: Choose and Setup Cloud Platform

#### Option A: AWS EC2 (Detailed Instructions)

**1. Launch EC2 Instance:**
   - Go to AWS Console → EC2 → Launch Instance
   - Choose AMI: "Deep Learning AMI (Ubuntu 20.04)" (has PyTorch pre-installed)
   - Instance type: `p3.2xlarge` (V100 GPU) or `g4dn.2xlarge` (T4 GPU, cheaper)
   - Storage: 100GB EBS volume
   - Security group: Allow SSH (port 22)

**2. Connect to instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

**3. Setup environment:**
   ```bash
   # Clone your repository
   git clone https://github.com/yourusername/fever-baseline.git
   cd fever-baseline

   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

   # Verify GPU is available
   python -c "import torch; print('GPU available:', torch.cuda.is_available())"
   ```

**4. Run the full pipeline:**
   ```bash
   # Run in a screen/tmux session to avoid disconnection issues
   screen -S fever_training

   # Run the pipeline
   python run.py all

   # Detach from screen: Ctrl+A then D
   # Re-attach later: screen -r fever_training
   ```

#### Option B: Google Cloud Compute Engine

**1. Create VM instance:**
   ```bash
   gcloud compute instances create fever-training \
     --zone=us-central1-a \
     --machine-type=n1-highmem-8 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --boot-disk-size=100GB \
     --maintenance-policy=TERMINATE
   ```

**2. SSH and setup:**
   ```bash
   gcloud compute ssh fever-training --zone=us-central1-a

   # Clone and setup (same as AWS above)
   git clone https://github.com/yourusername/fever-baseline.git
   cd fever-baseline
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

#### Option C: Paperspace Gradient

**1. Create a new project:**
   - Go to console.paperspace.com
   - Create new project
   - Create Notebook with PyTorch runtime
   - Choose GPU instance (P4000 or better)

**2. Upload your code:**
   - Use Jupyter interface to upload files
   - Or clone from git in terminal

**3. Run in terminal:**
   ```bash
   cd /notebooks/fever-baseline
   pip install -r requirements.txt
   python run.py all
   ```

### Step 4: Monitor Training

**Using the monitoring script:**
```bash
# In a separate terminal/screen session
python monitor_training.py
```

**Or check manually:**
```bash
# Check training progress
tail -f training.log  # if you redirect output

# Check GPU usage
nvidia-smi  # shows GPU utilization

# Check disk space
df -h

# Check running processes
ps aux | grep python
```

### Step 5: Download Results

**After training completes:**

```bash
# From your local machine, copy the trained model
scp -i your-key.pem ubuntu@your-instance-ip:~/fever-baseline/models/best_model.pt ./

# Or copy all results
scp -r -i your-key.pem ubuntu@your-instance-ip:~/fever-baseline/models ./
```

## Expected Timeline (with GPU)

| Stage | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| Data download | 5-10 min | 5-10 min |
| Wikipedia download | 15-30 min | 15-30 min |
| Wikipedia processing | 30-60 min | 2-4 hours |
| TF-IDF vectorization | 10-20 min | 1-2 hours |
| Model training (10 epochs) | 1-2 hours | 12-24 hours |
| Evaluation | 10-20 min | 1-2 hours |
| **TOTAL** | **~3-5 hours** | **~20-35 hours** |

## Performance Improvements vs Local

**With GPU (e.g., T4 or V100):**
- Training speed: 10-20x faster
- Total time: 3-5 hours vs 20-35 hours
- Full FEVER score: Actual retrieval vs oracle mode
- Better accuracy: Full pipeline evaluation

**Expected Results (from FEVER 2018 paper):**
- Label Accuracy: ~68-70%
- FEVER Score: ~49-52%
- Evidence Recall: ~70-75%

## Cost Estimation

**For complete training run (~5 hours on GPU):**
- AWS p3.2xlarge: ~$15
- Google Cloud n1-highmem-8 + T4: ~$3
- Paperspace P4000: ~$2.25

**Recommendations:**
1. Use spot/preemptible instances for 60-70% cost savings
2. Stop instance immediately after training
3. Use cloud storage (S3/GCS) for data to avoid re-downloading

## Troubleshooting

### Out of Memory (OOM) Errors
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # Instead of 32
```

### Slow Wikipedia Processing
```python
# Use multiprocessing (already implemented)
# Or use SSD storage instead of HDD
```

### Session Disconnection
```bash
# Always use screen or tmux
screen -S fever
python run.py all
# Ctrl+A then D to detach
```

### Check GPU Usage
```bash
# Monitor GPU every 2 seconds
watch -n 2 nvidia-smi
```

## After Training

**Evaluate your model:**
```bash
python run.py evaluate
```

**Expected output:**
```
================================================================================
Evaluation Results
================================================================================

Label Accuracy: 68.42%

FEVER Score: 51.23%

Evidence Retrieval:
  Precision: 45.67%
  Recall: 72.34%
  F1: 56.12%
```

## Cleanup

**Don't forget to:**
1. Download your trained model
2. Stop/terminate cloud instance
3. Delete large data files if not needed
4. Check billing to ensure no running instances

```bash
# AWS - terminate instance
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0

# Google Cloud - delete instance
gcloud compute instances delete fever-training --zone=us-central1-a
```

## Questions?

If you encounter issues:
1. Check logs in the `models/` directory
2. Verify GPU is detected: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check disk space: `df -h`
4. Monitor memory: `free -h`
