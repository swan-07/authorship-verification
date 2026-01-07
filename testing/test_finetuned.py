import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import pickle
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

print("=" * 80)
print("TESTING FINE-TUNED BERT")
print("=" * 80)

# Load dataset
print("\nLoading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

val_data = dataset['validation']
test_data = dataset['test']

# Use small sample for quick test
SAMPLE_SIZE = 500
test_data = test_data.select(range(SAMPLE_SIZE))

print(f"Validation: {len(val_data):,} pairs")
print(f"Test sample: {SAMPLE_SIZE} pairs")

# Load fine-tuned model
print("\nLoading fine-tuned model...")
model = SentenceTransformer('./bert-finetuned-authorship')
print("Model loaded!")

# Compute validation scores
print("\n1. Computing validation scores...")
val_scores = []
for i, ex in enumerate(val_data):
    emb1 = model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = model.encode(ex['text2'], convert_to_numpy=True)

    # Cosine similarity
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    val_scores.append(cos_sim)

    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(val_data)}")

val_scores = np.array(val_scores).reshape(-1, 1)
val_labels = np.array([ex['same'] for ex in val_data])

print("  Validation scores computed!")

# Train or load calibration
calibration_file = 'finetuned_calibration.pkl'
threshold_file = 'finetuned_threshold.txt'

if os.path.exists(calibration_file) and os.path.exists(threshold_file):
    print("\n2. Loading saved calibration...")
    with open(calibration_file, 'rb') as f:
        calibrator = pickle.load(f)
    with open(threshold_file, 'r') as f:
        optimal_threshold = float(f.read().strip())
    print(f"  Loaded! Threshold: {optimal_threshold:.4f}")
else:
    print("\n2. Training calibration on validation...")
    calibrator = LogisticRegression()
    calibrator.fit(val_scores, val_labels)

    # Find optimal threshold
    val_probs = calibrator.predict_proba(val_scores)[:, 1]
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    best_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[best_idx]

    # Save calibration
    with open(calibration_file, 'wb') as f:
        pickle.dump(calibrator, f)
    with open(threshold_file, 'w') as f:
        f.write(str(optimal_threshold))

    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Saved calibration to {calibration_file}")

# Compute test scores
print("\n3. Computing test scores...")
test_scores = []
for i, ex in enumerate(test_data):
    emb1 = model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = model.encode(ex['text2'], convert_to_numpy=True)

    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    test_scores.append(cos_sim)

    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(test_data)}")

test_scores = np.array(test_scores).reshape(-1, 1)
test_labels = np.array([ex['same'] for ex in test_data])

print("  Test scores computed!")

# Apply calibration and predict
print("\n4. Computing calibrated predictions...")
test_probs = calibrator.predict_proba(test_scores)[:, 1]
predictions = (test_probs >= optimal_threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
auc = roc_auc_score(test_labels, test_probs)

print("\n" + "=" * 80)
print("RESULTS: FINE-TUNED BERT + CALIBRATION")
print("=" * 80)
print(f"Threshold: {optimal_threshold:.4f}")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON TO BASELINE")
print("=" * 80)
print("Base BERT + Calibration:       Acc=63.7%, Prec=68.6%, Rec=50.5%, F1=58.2%")
print(f"Fine-tuned BERT + Calibration: Acc={accuracy*100:.1f}%, Prec={precision*100:.1f}%, Rec={recall*100:.1f}%, F1={f1*100:.1f}%")

improvement = (accuracy - 0.637) * 100
print(f"\nImprovement: {improvement:+.1f} percentage points")

# Save results
print("\n5. Saving results...")
np.savez('finetuned_test_results.npz',
         scores=test_scores.flatten(),
         probs=test_probs,
         predictions=predictions,
         labels=test_labels,
         threshold=optimal_threshold)
print("  Saved: finetuned_test_results.npz")

print("\nDone!")
