# Cloud Deployment Checklist

Before deploying to cloud, follow this checklist to ensure everything is ready.

## Pre-Deployment Checklist

### 1. Enable Wikipedia Download
- [ ] Open [config.py](config.py)
- [ ] Change `DOWNLOAD_WIKIPEDIA = False` to `DOWNLOAD_WIKIPEDIA = True`
- [ ] Save the file

### 2. Prepare Your Code Repository

**Option A: Using Git (Recommended)**

```bash
# Initialize git repository
cd c:\Users\Younes\Desktop\fever_code
git init

# Add all files (gitignore already configured)
git add .

# Commit
git commit -m "FEVER baseline ready for cloud deployment"

# Push to GitHub/GitLab
git remote add origin https://github.com/yourusername/fever-baseline.git
git push -u origin main
```

**Option B: Manual Upload**

- [ ] Zip the `fever_code` folder (exclude `.venv` and `data` folders)
- [ ] Upload to Google Drive or your cloud platform
- [ ] Remember the download link or location

### 3. Choose Cloud Platform

Select one platform:

- [ ] **AWS EC2** - Most flexible, pay-per-hour (~$0.60-3/hour)
- [ ] **Google Cloud** - Good free credits for new users (~$0.60/hour)
- [ ] **Paperspace Gradient** - ML-focused, easiest setup (~$0.45/hour)
- [ ] **Google Colab Pro+** - Jupyter interface, monthly subscription ($50/month)

### 4. Verify Files to Upload

Make sure these files are in your repository:

**Core files (required):**
- [x] config.py (with DOWNLOAD_WIKIPEDIA=True)
- [x] requirements.txt
- [x] run.py
- [x] download_data.py
- [x] wiki_processor.py
- [x] retrieval.py
- [x] model.py
- [x] train.py
- [x] evaluate.py
- [x] fever_scorer.py
- [x] data_loader.py
- [x] tokenizer.py
- [x] dataset.py

**Helper files:**
- [x] setup_cloud.sh
- [x] monitor_training.py
- [x] CLOUD_DEPLOYMENT.md
- [x] README.md
- [x] .gitignore

**DO NOT upload:**
- [ ] .venv/ folder (virtual environment)
- [ ] data/ folder (will be downloaded on cloud)
- [ ] models/ folder (will be created during training)

## Deployment Steps

### Quick Start (Copy-Paste Commands)

Once you're on your cloud instance, run these commands:

```bash
# Clone your repository
git clone https://github.com/yourusername/fever-baseline.git
cd fever-baseline

# Run setup script
bash setup_cloud.sh

# Start training in screen session
screen -S fever
python run.py all

# Detach from screen (Ctrl+A then D)
# You can now disconnect from SSH without stopping training

# To monitor progress, reconnect and run:
screen -r fever  # Reattach to see training output
# OR
python monitor_training.py  # Monitor without interrupting
```

## Post-Deployment Checklist

### During Training
- [ ] Verify GPU is being used: `nvidia-smi`
- [ ] Check disk space: `df -h`
- [ ] Monitor training progress: `python monitor_training.py`
- [ ] Verify Wikipedia download completed (~50GB)

### After Training
- [ ] Download trained model: `models/best_model.pt`
- [ ] Save evaluation results
- [ ] **IMPORTANT**: Stop/terminate cloud instance to avoid charges
- [ ] Verify instance is fully terminated in cloud console

## Commands Reference

### Screen Session Management
```bash
# Create new screen session
screen -S fever

# Detach from screen
Ctrl+A then D

# List all screen sessions
screen -ls

# Reattach to session
screen -r fever

# Kill screen session
screen -X -S fever quit
```

### Monitor Training
```bash
# Real-time monitoring
python monitor_training.py

# Check GPU usage
watch -n 2 nvidia-smi

# Check disk space
df -h

# Check running processes
ps aux | grep python

# View last 50 lines of training output
tail -n 50 training.log  # if logging to file
```

### Download Results (from local machine)
```bash
# Download trained model
scp -i your-key.pem ubuntu@your-instance-ip:~/fever-baseline/models/best_model.pt ./

# Download all models
scp -r -i your-key.pem ubuntu@your-instance-ip:~/fever-baseline/models ./

# Download entire project
scp -r -i your-key.pem ubuntu@your-instance-ip:~/fever-baseline ./fever-results
```

## Expected Timeline (with T4 GPU)

| Task | Estimated Time |
|------|---------------|
| Instance setup | 5-10 minutes |
| Code upload/clone | 1-2 minutes |
| Dependencies installation | 2-3 minutes |
| Data download (train/dev/test) | 5-10 minutes |
| Wikipedia download | 15-30 minutes |
| Wikipedia processing | 30-60 minutes |
| TF-IDF vectorization | 10-20 minutes |
| Model training (10 epochs) | 1-2 hours |
| Evaluation | 10-20 minutes |
| **TOTAL** | **~3-5 hours** |

## Cost Estimation

### One complete training run (~5 hours)

| Platform | Instance Type | Cost per Hour | Total Cost |
|----------|--------------|---------------|------------|
| AWS EC2 | g4dn.2xlarge (T4) | $0.752 | ~$3.76 |
| AWS EC2 (spot) | g4dn.2xlarge | ~$0.23 | ~$1.15 |
| Google Cloud | n1-highmem-8 + T4 | $0.60 | ~$3.00 |
| Google Cloud (preemptible) | n1-highmem-8 + T4 | ~$0.20 | ~$1.00 |
| Paperspace | P4000 | $0.45 | ~$2.25 |

**💡 Tip**: Use spot/preemptible instances for 60-70% cost savings!

## Troubleshooting

### Issue: Out of disk space
```bash
# Check disk usage
df -h

# Clean pip cache
pip cache purge

# Solution: Use larger instance storage (100GB recommended)
```

### Issue: Out of memory (OOM)
```python
# Edit config.py
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: CUDA out of memory
```python
# Edit config.py
BATCH_SIZE = 16  # or even 8
```

### Issue: Training disconnected
```bash
# Always use screen or tmux!
# Check if process is still running:
ps aux | grep python

# Reattach to screen:
screen -r fever
```

### Issue: Wikipedia download failed
```bash
# Manually retry download
cd data
wget https://fever.ai/download/fever/wiki-pages.zip
unzip wiki-pages.zip
```

## Final Reminder

**Before you disconnect:**
1. ✓ Training is running in a screen session
2. ✓ You can detach without killing the process (Ctrl+A then D)
3. ✓ You know how to reconnect: `screen -r fever`

**After training completes:**
1. ✓ Download your trained model
2. ✓ Download evaluation results
3. ✓ **TERMINATE THE INSTANCE** to stop charges
4. ✓ Verify termination in cloud console

---

**Good luck with your cloud deployment! 🚀**

For detailed instructions, see [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md)
