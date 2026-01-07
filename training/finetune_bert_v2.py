import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# MUST set this BEFORE importing torch to disable MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import pickle
import torch
# Explicitly disable MPS and force CPU
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

print("=" * 80)
print("FINE-TUNING BERT V2 (CPU-ONLY)")
print("=" * 80)

device = torch.device("cpu")  # Force CPU
print(f"\nDevice: {device} (MPS disabled, using CPU only)")

# Load cached dataset
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use smaller subset for faster iteration
TRAIN_SIZE = 50000  # Use 50k instead of 325k for faster training
print(f"\nUsing {TRAIN_SIZE:,} training examples (subset for speed)")

train_data = dataset['train']
print(f"  Converting to InputExamples...")

train_examples = []
for i in range(min(TRAIN_SIZE, len(train_data))):
    ex = train_data[i]
    score = float(ex['same'])
    train_examples.append(InputExample(texts=[ex['text1'], ex['text2']], label=score))

    if i % 10000 == 0:
        print(f"    {i}/{TRAIN_SIZE}")

print(f"  Created {len(train_examples):,} training pairs")

# Load model
print("\nLoading bert-base-cased model...")
model = SentenceTransformer('bert-base-cased')
print(f"  Model loaded successfully")
print(f"  Max sequence length: {model.max_seq_length}")

# Create DataLoader with smaller batch size for memory
print("\nCreating DataLoader...")
BATCH_SIZE = 4  # Reduce from 16 to 4 to avoid OOM
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Total batches: {len(train_dataloader):,}")

# Use CosineSimilarityLoss
print("\nSetting up loss function...")
train_loss = losses.CosineSimilarityLoss(model)
print(f"  Using CosineSimilarityLoss")

# Training parameters
num_epochs = 1
warmup_steps = int(len(train_dataloader) * 0.1)

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Training examples: {len(train_examples):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {num_epochs}")
print(f"Steps per epoch: {len(train_dataloader):,}")
print(f"Warmup steps: {warmup_steps:,}")
print(f"Output: ./bert-finetuned-authorship")

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

try:
    # Train with explicit parameters
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path='./bert-finetuned-authorship',
        show_progress_bar=True,
        use_amp=False  # Disable automatic mixed precision to avoid issues
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Model saved to: ./bert-finetuned-authorship")

except Exception as e:
    print(f"\n!!! TRAINING FAILED !!!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
