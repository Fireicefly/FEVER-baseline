# Quick Start - Cloud Deployment

Choose your platform and follow the copy-paste commands below.

## Before You Start

**1. Enable Wikipedia download:**
```python
# Edit config.py and change:
DOWNLOAD_WIKIPEDIA = True
```

**2. Upload your code to GitHub:**
```bash
cd c:\Users\Younes\Desktop\fever_code
git init
git add .
git commit -m "Ready for cloud"
git remote add origin https://github.com/YOUR_USERNAME/fever-baseline.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## Option 1: AWS EC2 (Best Performance/Cost)

### Step 1: Launch Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose: **Deep Learning AMI (Ubuntu 20.04)**
3. Instance type: **g4dn.2xlarge** (T4 GPU - recommended)
4. Storage: **100GB**
5. Security group: Allow SSH (port 22)
6. Launch and download your key pair (.pem file)

### Step 2: Connect

```bash
# Make key private (Linux/Mac)
chmod 400 your-key.pem

# Connect to instance
ssh -i your-key.pem ubuntu@YOUR_INSTANCE_IP
```

Replace `YOUR_INSTANCE_IP` with the public IP from AWS console.

### Step 3: Setup and Run

Copy-paste these commands on your instance:

```bash
# Clone your code
git clone https://github.com/YOUR_USERNAME/fever-baseline.git
cd fever-baseline

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Start training in screen
screen -S fever
python run.py all

# Press Ctrl+A then D to detach (training continues in background)
```

### Step 4: Monitor (Optional)

In a new terminal:
```bash
ssh -i your-key.pem ubuntu@YOUR_INSTANCE_IP
cd fever-baseline
source .venv/bin/activate
python monitor_training.py
```

### Step 5: Download Results

When done, from your local machine:
```bash
scp -i your-key.pem ubuntu@YOUR_INSTANCE_IP:~/fever-baseline/models/best_model.pt ./
```

### Step 6: Cleanup

**IMPORTANT**: Terminate instance to stop charges!
```bash
# In AWS Console: EC2 → Instances → Select instance → Instance State → Terminate
```

**Cost**: ~$0.75/hour = ~$3.75 for 5 hours

---

## Option 2: Google Cloud (Best for Beginners)

### Step 1: Setup Project

```bash
# Install Google Cloud CLI if not installed
# https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Create or select project
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Create Instance

```bash
# Create VM with T4 GPU
gcloud compute instances create fever-training \
  --zone=us-central1-a \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

### Step 3: Connect

```bash
gcloud compute ssh fever-training --zone=us-central1-a
```

### Step 4: Setup and Run

```bash
# Clone your code
git clone https://github.com/YOUR_USERNAME/fever-baseline.git
cd fever-baseline

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Start training in screen
screen -S fever
python run.py all

# Press Ctrl+A then D to detach
```

### Step 5: Download Results

From your local machine:
```bash
gcloud compute scp fever-training:~/fever-baseline/models/best_model.pt ./ --zone=us-central1-a
```

### Step 6: Cleanup

```bash
gcloud compute instances delete fever-training --zone=us-central1-a
```

**Cost**: ~$0.60/hour = ~$3.00 for 5 hours

---

## Option 3: Paperspace Gradient (Easiest)

### Step 1: Sign Up

1. Go to https://console.paperspace.com
2. Create account
3. Add payment method

### Step 2: Create Notebook

1. Create New Project
2. Select "PyTorch" runtime
3. Choose machine: **P4000** or **P5000** (GPU)
4. Start notebook

### Step 3: Setup in Notebook Terminal

Click on "Terminal" in Jupyter interface, then run:

```bash
# Clone your code
cd /notebooks
git clone https://github.com/YOUR_USERNAME/fever-baseline.git
cd fever-baseline

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Start training
screen -S fever
python run.py all

# Press Ctrl+A then D to detach
```

### Step 4: Monitor

In Jupyter terminal:
```bash
cd /notebooks/fever-baseline
python monitor_training.py
```

### Step 5: Download Results

Use Jupyter file browser to download `models/best_model.pt`

### Step 6: Cleanup

**IMPORTANT**: Stop the notebook to avoid charges!

1. Go to Gradient Console
2. Stop the notebook

**Cost**: ~$0.45/hour = ~$2.25 for 5 hours

---

## Comparison Table

| Platform | Setup Difficulty | Cost (5 hours) | GPU | Good For |
|----------|-----------------|----------------|-----|----------|
| **AWS EC2** | Medium | ~$3.75 | T4 | Best price/performance |
| **Google Cloud** | Medium | ~$3.00 | T4 | New users (free credits) |
| **Paperspace** | Easy | ~$2.25 | P4000/P5000 | Beginners, Jupyter lovers |

---

## Universal Commands (All Platforms)

### Monitor Training

```bash
# Method 1: Monitoring script
python monitor_training.py

# Method 2: Check GPU
watch -n 2 nvidia-smi

# Method 3: Reattach to screen
screen -r fever
```

### Common Issues

**GPU not detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install driver (varies by platform)
```

**Out of memory:**
```python
# Edit config.py, reduce batch size
BATCH_SIZE = 16  # or 8
```

**Disk full:**
```bash
# Check space
df -h

# Clean cache
pip cache purge
```

---

## Expected Results

After ~5 hours, you should have:

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

================================================================================
```

---

## Next Steps After Training

1. **Download your model**
2. **Run additional evaluation**:
   ```bash
   python run.py evaluate --data data/test.jsonl
   ```
3. **Analyze results**
4. **Share your findings!**

---

## Need Help?

- Check [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for detailed guide
- Check [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) for checklist
- Check [README.md](README.md) for project overview

---

**Happy training! 🚀**
