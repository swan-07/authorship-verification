#!/usr/bin/env python3
"""Fine-tune BERT for authorship verification using sentence-transformers."""

import sys
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

print("=" * 80)
print("FINE-TUNING BERT FOR AUTHORSHIP VERIFICATION")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
dataset = load_dataset('swan07/authorship-verification')
train_data = dataset['train']
val_data = dataset.get('validation') or dataset.get('val') or dataset['train'].select(range(1000))
print(f"   Train: {len(train_data)} samples")
print(f"   Val: {len(val_data)} samples")

# Create training examples
print("\n2. Preparing training data...")
train_examples = []
for i, ex in enumerate(train_data):
    if i % 5000 == 0:
        print(f"   {i}/{len(train_data)}...")
    train_examples.append(InputExample(
        texts=[ex['text1'], ex['text2']],
        label=float(ex['same'])
    ))

# Load base model
print("\n3. Loading base BERT model...")
model = SentenceTransformer('bert-base-cased')

# Create dataloader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss
train_loss = losses.CosineSimilarityLoss(model)

# Prepare evaluator
print("\n4. Preparing validation evaluator...")
val_samples = []
val_labels = []
for i, ex in enumerate(val_data):
    if i >= 1000:  # Use 1000 samples for validation
        break
    val_samples.append(InputExample(texts=[ex['text1'], ex['text2']], label=float(ex['same'])))

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_samples, name='val')

# Train
print("\n5. Training...")
print(f"   Epochs: 3")
print(f"   Batch size: 16")
print(f"   Training samples: {len(train_examples)}")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=1000,
    warmup_steps=500,
    output_path='./finetuned_bert_model',
    save_best_model=True,
    show_progress_bar=True
)

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE!")
print("=" * 80)
print("\nModel saved to: ./finetuned_bert_model")
