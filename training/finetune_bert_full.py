import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import torch
# GPU (MPS) enabled for faster training!

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

print("=" * 80)
print("FULL BERT FINE-TUNING ON ALL 325K+ TRAINING PAIRS")
print("With Checkpointing Every 10K Steps")
print("=" * 80)

# Configuration
CHECKPOINT_DIR = 'bert_full_checkpoints'
MODEL_NAME = 'bert-finetuned-authorship-full'
BATCH_SIZE = 4  # Safe for GPU memory
EPOCHS = 1
CHECKPOINT_EVERY = 10000  # Save every 10K training steps

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use 100K pairs - good balance for training time and performance
train_data = dataset['train'].select(range(100000))
val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(1000))

print(f"Training: {len(train_data):,} pairs")
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Check for existing checkpoint
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.txt')
start_idx = 0
model = None

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint_path = f.read().strip()

    if os.path.exists(checkpoint_path):
        print(f"\n2. RESUMING from checkpoint: {checkpoint_path}")
        model = SentenceTransformer(checkpoint_path)

        # Extract start index from checkpoint name
        checkpoint_num = int(checkpoint_path.split('_')[-1].replace('checkpoint', ''))
        start_idx = checkpoint_num

        print(f"   Resuming from pair {start_idx:,}")
    else:
        print(f"\n2. Checkpoint file not found, starting fresh")
else:
    print("\n2. No checkpoint found, starting fresh")

if model is None:
    print("\n3. Loading base BERT model...")
    model = SentenceTransformer('bert-base-cased')

# Prepare training examples with flush for progress visibility
print(f"\n4. Preparing training examples from index {start_idx:,}...", flush=True)

train_examples = []
for i in range(start_idx, len(train_data)):
    ex = train_data[i]

    # Label: 1 if same author, 0 if different
    # For CosineSimilarityLoss: high similarity score (label=1) for same author
    train_examples.append(InputExample(
        texts=[ex['text1'], ex['text2']],
        label=float(ex['same'])
    ))

    if (i + 1 - start_idx) % 10000 == 0:
        print(f"   Prepared {i+1:,}/{len(train_data):,} examples", flush=True)

print(f"   ✓ Prepared {len(train_examples):,} training examples", flush=True)

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# Define loss
train_loss = losses.CosineSimilarityLoss(model)

# Calculate total steps
steps_per_epoch = len(train_dataloader)
total_steps = steps_per_epoch * EPOCHS

print(f"\n5. Training configuration:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Steps per epoch: {steps_per_epoch:,}")
print(f"   Total training steps: {total_steps:,}")
print(f"   Checkpoint every: {CHECKPOINT_EVERY:,} steps")

# Custom callback for checkpointing
class CheckpointCallback:
    def __init__(self, model, checkpoint_dir, checkpoint_every, start_idx):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.global_step = 0
        self.start_idx = start_idx

    def __call__(self, score, epoch, steps):
        self.global_step = steps

        # Calculate actual training pair index
        actual_idx = self.start_idx + (steps * BATCH_SIZE)

        if steps > 0 and steps % self.checkpoint_every == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_{actual_idx}'
            )
            self.model.save(checkpoint_path)

            # Update latest checkpoint pointer
            with open(os.path.join(self.checkpoint_dir, 'latest_checkpoint.txt'), 'w') as f:
                f.write(checkpoint_path)

            print(f"\n   ✓ Checkpoint saved at step {steps:,} (pair {actual_idx:,})")
            print(f"   Saved to: {checkpoint_path}")

# Train
print(f"\n6. Starting training...")
print("   (This will take several hours - checkpoints will be saved regularly)")

start_time = time.time()

callback = CheckpointCallback(model, CHECKPOINT_DIR, CHECKPOINT_EVERY, start_idx)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    output_path=MODEL_NAME,
    show_progress_bar=True,
    callback=callback,
    checkpoint_path=CHECKPOINT_DIR,
    checkpoint_save_steps=CHECKPOINT_EVERY,
    checkpoint_save_total_limit=3  # Keep only last 3 checkpoints to save space
)

elapsed = time.time() - start_time
print(f"\n   ✓ Training complete in {elapsed/3600:.1f} hours")

# Save final model
print(f"\n7. Saving final model to {MODEL_NAME}/...")
model.save(MODEL_NAME)
print("   ✓ Model saved")

# Evaluate on validation set
print("\n8. Evaluating on validation set...")

val_embeddings_1 = []
val_embeddings_2 = []
val_labels = []

for ex in val_data:
    emb1 = model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = model.encode(ex['text2'], convert_to_numpy=True)

    val_embeddings_1.append(emb1)
    val_embeddings_2.append(emb2)
    val_labels.append(ex['same'])

val_embeddings_1 = np.array(val_embeddings_1)
val_embeddings_2 = np.array(val_embeddings_2)
val_labels = np.array(val_labels)

# Compute cosine similarities
val_sims = []
for i in range(len(val_embeddings_1)):
    cos_sim = np.dot(val_embeddings_1[i], val_embeddings_2[i]) / (
        np.linalg.norm(val_embeddings_1[i]) * np.linalg.norm(val_embeddings_2[i])
    )
    val_sims.append(cos_sim)

val_sims = np.array(val_sims)

# Train calibrator
print("\n9. Training calibration model...")
calibrator = LogisticRegression(random_state=42)
calibrator.fit(val_sims.reshape(-1, 1), val_labels)

with open('finetuned_full_calibration.pkl', 'wb') as f:
    pickle.dump(calibrator, f)

print("   ✓ Calibration model saved")

# Get optimal threshold
val_probs = calibrator.predict_proba(val_sims.reshape(-1, 1))[:, 1]
thresholds = np.arange(0, 1, 0.01)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Optimal threshold: {best_threshold:.4f} (F1={best_f1*100:.1f}%)")

# Test on test set
print("\n10. Testing on test set...")

test_sims = []
test_labels = []

for ex in test_data:
    emb1 = model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = model.encode(ex['text2'], convert_to_numpy=True)

    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    test_sims.append(cos_sim)
    test_labels.append(ex['same'])

test_sims = np.array(test_sims)
test_labels = np.array(test_labels)

test_probs = calibrator.predict_proba(test_sims.reshape(-1, 1))[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
auc = roc_auc_score(test_labels, test_probs)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"Training size: {len(train_data):,} pairs")
print(f"Test size: {len(test_data):,} pairs")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("BERT (50K pairs):    Acc=70.1%, F1=71.6%, AUC=0.760")
print(f"BERT (325K pairs):   Acc={accuracy*100:.1f}%, F1={f1*100:.1f}%, AUC={auc:.3f}")

improvement = (accuracy - 0.701) * 100
print(f"\nImprovement: {improvement:+.1f} percentage points")

print("\n✓ Done!")
print(f"Model saved to: {MODEL_NAME}/")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
