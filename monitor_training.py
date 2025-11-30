"""
Monitor FEVER training progress.

This script checks the training status every 30 seconds.
"""

import os
import time
from datetime import datetime
import subprocess


def get_file_size(filepath):
    """Get file size in human-readable format."""
    if not os.path.exists(filepath):
        return "N/A"

    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def check_process_running(process_name="python"):
    """Check if a Python process is running."""
    try:
        # Windows command
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        return process_name.lower() in result.stdout.lower()
    except:
        return False


def monitor_training():
    """Monitor training progress."""
    print("=" * 80)
    print("FEVER Training Monitor")
    print("Checking status every 30 seconds... (Press Ctrl+C to stop)")
    print("=" * 80)
    print()

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n[{timestamp}] Check #{iteration}")
            print("-" * 80)

            # Check if Python is running
            python_running = check_process_running("python")
            print(f"Python process running: {'YES' if python_running else 'NO'}")

            # Check data files
            print("\nData Files:")
            print(f"  train.jsonl: {get_file_size('data/train.jsonl')}")
            print(f"  dev.jsonl: {get_file_size('data/dev.jsonl')}")
            print(f"  test.jsonl: {get_file_size('data/test.jsonl')}")

            # Check model files
            print("\nModel Files:")
            print(f"  vocab.pkl: {get_file_size('models/vocab.pkl')}")
            print(f"  best_model.pt: {get_file_size('models/best_model.pt')}")

            # Check if training is complete
            if os.path.exists('models/best_model.pt'):
                print("\n" + "=" * 80)
                print("TRAINING COMPLETE!")
                print("=" * 80)
                print("\nModel saved at: models/best_model.pt")
                print("You can now run evaluation with:")
                print("  python run.py evaluate")
                break

            # Check directory contents
            if os.path.exists('models'):
                models_files = os.listdir('models')
                if models_files:
                    print(f"\nFiles in models/: {', '.join(models_files)}")

            print("\nWaiting 30 seconds for next check...")
            print("-" * 80)

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("=" * 80)


if __name__ == "__main__":
    # Change to project directory
    os.chdir(r"c:\Users\Younes\Desktop\fever_code")
    monitor_training()
