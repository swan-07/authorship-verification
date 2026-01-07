import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Disable TensorFlow to avoid tf-keras dependency
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import pickle
import torch
# Explicitly disable MPS and force CPU
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import json
from pathlib import Path

print("=" * 80)
print("FINE-TUNING BERT V3 (WITH CHECKPOINTING)")
print("=" * 80)

device = torch.device("cpu")
print(f"\nDevice: {device} (MPS disabled, using CPU only)")

# Checkpoint directory
CHECKPOINT_DIR = "./bert-finetuned-authorship-checkpoints"
FINAL_MODEL_DIR = "./bert-finetuned-authorship"
STATE_FILE = os.path.join(CHECKPOINT_DIR, "training_state.json")

# Load cached dataset
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Training parameters
TRAIN_SIZE = 50000
BATCH_SIZE = 4
num_epochs = 1

# Prepare training data
print(f"\nPreparing training data (using {TRAIN_SIZE:,} examples)...")
train_data = dataset['train']
train_examples = []

for i in range(min(TRAIN_SIZE, len(train_data))):
    ex = train_data[i]
    score = float(ex['same'])
    train_examples.append(InputExample(texts=[ex['text1'], ex['text2']], label=score))

    if i % 10000 == 0:
        print(f"    {i}/{TRAIN_SIZE}")

print(f"  Created {len(train_examples):,} training pairs")

# Create DataLoader
print("\nCreating DataLoader...")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total batches: {len(train_dataloader):,}")

# Check for existing checkpoint
resume_from = None
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith('checkpoint-')]
    if checkpoints:
        # Find latest checkpoint
        checkpoint_nums = [int(c.split('-')[1]) for c in checkpoints]
        latest_checkpoint_num = max(checkpoint_nums)
        resume_from = os.path.join(CHECKPOINT_DIR, f'checkpoint-{latest_checkpoint_num}')
        print(f"\n✓ Found checkpoint: {resume_from}")
        print(f"  Resuming from step {latest_checkpoint_num}")
    else:
        print("\nNo checkpoint found, starting from scratch")
else:
    print("\nNo checkpoint found, starting from scratch")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load or create model
if resume_from and os.path.exists(resume_from):
    print(f"\nLoading model from checkpoint: {resume_from}")
    model = SentenceTransformer(resume_from)
    print("  ✓ Model loaded from checkpoint")
else:
    print("\nLoading bert-base-cased model...")
    model = SentenceTransformer('bert-base-cased')
    print(f"  Model loaded successfully")

print(f"  Max sequence length: {model.max_seq_length}")

# Use CosineSimilarityLoss
print("\nSetting up loss function...")
train_loss = losses.CosineSimilarityLoss(model)

# Training parameters with checkpointing
warmup_steps = int(len(train_dataloader) * 0.1)
save_steps = 500  # Save every 500 steps
logging_steps = 100  # Log every 100 steps

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Training examples: {len(train_examples):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {num_epochs}")
print(f"Steps per epoch: {len(train_dataloader):,}")
print(f"Warmup steps: {warmup_steps:,}")
print(f"Save checkpoint every: {save_steps} steps")
print(f"Logging every: {logging_steps} steps")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Final model directory: {FINAL_MODEL_DIR}")
if resume_from:
    print(f"Resuming from: {resume_from}")

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print("Training will save checkpoints every 500 steps.")
print("If interrupted, restart this script to resume from last checkpoint.\n")

try:
    # Train with checkpointing enabled
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=FINAL_MODEL_DIR,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_save_steps=save_steps,
        checkpoint_save_total_limit=3,  # Keep only last 3 checkpoints to save space
        show_progress_bar=True,
        use_amp=False
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final model saved to: {FINAL_MODEL_DIR}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")

    # Save completion marker
    with open(os.path.join(CHECKPOINT_DIR, "TRAINING_COMPLETE"), 'w') as f:
        f.write("Training completed successfully\n")

except KeyboardInterrupt:
    print("\n" + "=" * 80)
    print("TRAINING INTERRUPTED BY USER")
    print("=" * 80)
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("Restart this script to resume from last checkpoint.")

except Exception as e:
    print(f"\n!!! TRAINING FAILED !!!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nCheckpoints may be available in: {CHECKPOINT_DIR}")
    print("If checkpoints exist, restart this script to resume.")
