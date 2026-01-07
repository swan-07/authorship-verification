import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

print("=" * 80)
print("FINE-TUNING BERT FOR AUTHORSHIP VERIFICATION")
print("=" * 80)

# Load cached dataset
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use a subset for faster training
print("\nPreparing training data...")
train_data = dataset['train']
val_data = dataset['validation']

# Convert to InputExample format for fine-tuning
# For authorship verification, we use pairs with similarity scores (0 or 1)
train_examples = []
print(f"  Processing {len(train_data)} training examples...")
for i, ex in enumerate(train_data):
    # SentenceTransformer expects similarity score between 0-1
    score = float(ex['same'])  # 0 for different, 1 for same
    train_examples.append(InputExample(texts=[ex['text1'], ex['text2']], label=score))

    if i % 10000 == 0:
        print(f"    {i}/{len(train_data)}")

print(f"  Created {len(train_examples)} training pairs")

# Validation evaluator
print("\nPreparing validation evaluator...")
val_subset = val_data.select(range(5000))  # Use subset for faster eval
val_sentences1 = [ex['text1'] for ex in val_subset]
val_sentences2 = [ex['text2'] for ex in val_subset]
val_scores = [float(ex['same']) for ex in val_subset]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    val_sentences1,
    val_sentences2,
    val_scores,
    name='authorship-val'
)

# Load base model
print("\nLoading bert-base-cased model...")
model = SentenceTransformer('bert-base-cased')

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Use CosineSimilarityLoss for training on pairs
train_loss = losses.CosineSimilarityLoss(model)

# Training parameters
num_epochs = 1  # Start with 1 epoch to see results faster
warmup_steps = int(len(train_dataloader) * 0.1)  # 10% warmup

print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Training examples: {len(train_examples):,}")
print(f"Validation examples: {len(val_scores):,}")
print(f"Batch size: 16")
print(f"Epochs: {num_epochs}")
print(f"Steps per epoch: {len(train_dataloader):,}")
print(f"Warmup steps: {warmup_steps:,}")
print(f"Total training steps: {len(train_dataloader) * num_epochs:,}")

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print("This will take ~2-3 hours for 1 epoch on the full training set.")
print("Progress will be shown every 1000 steps.\n")

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluation_steps=5000,  # Evaluate every 5000 steps
    output_path='./models/bert-finetuned-authorship',
    save_best_model=True,
    show_progress_bar=True
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("Model saved to: ./models/bert-finetuned-authorship")
print("\nNext step: Test the fine-tuned model on the test set.")
