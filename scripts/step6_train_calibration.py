"""
Train Calibration Model for Base BERT
Converts raw cosine similarity scores to calibrated probabilities.
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import pickle

print("="*80)
print("TRAINING CALIBRATION MODEL FOR BASE BERT")
print("="*80)

# Load base BERT model
print("\n1. Loading base BERT model...")
try:
    model = SentenceTransformer('bert-base-cased')
    print(f"   ✓ Model loaded")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Load validation dataset (this is used for calibration training)
print("\n2. Loading validation dataset...")
try:
    val_dataset = load_dataset(
        "swan07/authorship-verification",
        data_files="*_val.csv",
        split='train'
    )
    print(f"   ✓ Loaded {len(val_dataset)} validation samples")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Extract cosine similarities on validation set
print("\n3. Computing cosine similarities on validation set...")
print(f"   Processing {len(val_dataset)} samples...")

cosine_scores = []
true_labels = []

# Use smaller sample if dataset is too large
max_samples = min(5000, len(val_dataset))
sample_indices = np.random.choice(len(val_dataset), max_samples, replace=False)

for idx in tqdm(sample_indices, desc="   Computing scores"):
    sample = val_dataset[int(idx)]
    text1 = sample['text1']
    text2 = sample['text2']

    # Determine label column name
    if 'same' in sample:
        label = sample['same']
    elif 'score' in sample:
        label = sample['score']
    else:
        print("   ✗ Cannot find label column!")
        sys.exit(1)

    # Get embeddings
    emb1 = model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(text2, convert_to_tensor=True, show_progress_bar=False)

    # Cosine similarity
    cos_sim = util.pytorch_cos_sim(emb1, emb2).item()

    cosine_scores.append(cos_sim)
    true_labels.append(label)

cosine_scores = np.array(cosine_scores).reshape(-1, 1)
true_labels = np.array(true_labels)

print(f"   ✓ Computed {len(cosine_scores)} scores")

# Analyze score distribution before calibration
print("\n4. Score distribution analysis:")
same_scores = cosine_scores[true_labels == 1].flatten()
diff_scores = cosine_scores[true_labels == 0].flatten()

print(f"   Same author (n={len(same_scores)}):")
print(f"     Mean: {np.mean(same_scores):.3f}")
print(f"     Range: {np.min(same_scores):.3f} - {np.max(same_scores):.3f}")

print(f"   Different author (n={len(diff_scores)}):")
print(f"     Mean: {np.mean(diff_scores):.3f}")
print(f"     Range: {np.min(diff_scores):.3f} - {np.max(diff_scores):.3f}")

# Train calibration model (Logistic Regression)
print("\n5. Training Logistic Regression calibration model...")

calibration_model = LogisticRegression(max_iter=1000, random_state=42)
calibration_model.fit(cosine_scores, true_labels)

print(f"   ✓ Model trained")
print(f"   Coefficients: {calibration_model.coef_[0]}")
print(f"   Intercept: {calibration_model.intercept_[0]}")

# Evaluate calibration on validation set
print("\n6. Evaluating calibration...")

# Get calibrated probabilities
calibrated_probs = calibration_model.predict_proba(cosine_scores)[:, 1]

# Test different thresholds
print("\n   Testing different probability thresholds:")
print("   " + "-"*70)
print(f"   {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("   " + "-"*70)

best_f1 = 0
best_threshold = 0.5

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (calibrated_probs >= thresh).astype(int)
    acc = accuracy_score(true_labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average='binary', zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

    print(f"   {thresh:<12.1f} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")

print("   " + "-"*70)
print(f"   Best threshold: {best_threshold:.1f} (F1={best_f1:.3f})")

# Final evaluation at best threshold
best_preds = (calibrated_probs >= best_threshold).astype(int)
final_acc = accuracy_score(true_labels, best_preds)
final_prec, final_rec, final_f1, _ = precision_recall_fscore_support(
    true_labels, best_preds, average='binary'
)

print(f"\n7. Final calibrated results (threshold={best_threshold}):")
print(f"   Accuracy:  {final_acc:.3f}")
print(f"   Precision: {final_prec:.3f}")
print(f"   Recall:    {final_rec:.3f}")
print(f"   F1 Score:  {final_f1:.3f}")

# Calculate AUC
try:
    auc = roc_auc_score(true_labels, calibrated_probs)
    print(f"   AUC:       {auc:.3f}")
except:
    pass

# Compare: Before vs After Calibration
print("\n8. Calibration Impact:")

# Before: Using raw cosine similarity with threshold 0.5
raw_preds = (cosine_scores.flatten() >= 0.5).astype(int)
raw_acc = accuracy_score(true_labels, raw_preds)
raw_prec, raw_rec, raw_f1, _ = precision_recall_fscore_support(
    true_labels, raw_preds, average='binary', zero_division=0
)

print(f"\n   BEFORE (raw cosine, threshold=0.5):")
print(f"     Accuracy:  {raw_acc:.3f}")
print(f"     Precision: {raw_prec:.3f}")
print(f"     Recall:    {raw_rec:.3f}")
print(f"     F1 Score:  {raw_f1:.3f}")

print(f"\n   AFTER (calibrated, threshold={best_threshold}):")
print(f"     Accuracy:  {final_acc:.3f}")
print(f"     Precision: {final_prec:.3f}")
print(f"     Recall:    {final_rec:.3f}")
print(f"     F1 Score:  {final_f1:.3f}")

improvement = final_f1 - raw_f1
print(f"\n   Improvement: +{improvement:.3f} F1 score")

# Save calibration model
print("\n9. Saving calibration model...")

model_path = "/Users/swan/Documents/GitHub/authorship-verification/calibration_model_base_bert.pkl"
joblib.dump(calibration_model, model_path)
print(f"   ✓ Saved to: {model_path}")

# Save config for later use
config = {
    'best_threshold': best_threshold,
    'model_type': 'LogisticRegression',
    'base_model': 'bert-base-cased',
    'training_samples': len(cosine_scores),
    'coefficients': calibration_model.coef_[0].tolist(),
    'intercept': calibration_model.intercept_[0],
    'performance': {
        'accuracy': final_acc,
        'precision': final_prec,
        'recall': final_rec,
        'f1': final_f1
    }
}

config_path = "/Users/swan/Documents/GitHub/authorship-verification/calibration_config.pkl"
with open(config_path, 'wb') as f:
    pickle.dump(config, f)
print(f"   ✓ Config saved to: {config_path}")

print("\n" + "="*80)
print("✓ CALIBRATION TRAINING COMPLETE!")
print("="*80)

print(f"""
Summary:
  Model: Logistic Regression
  Training samples: {len(cosine_scores)}
  Best threshold: {best_threshold}

  Performance:
    Accuracy:  {final_acc:.1%}
    Precision: {final_prec:.1%}
    Recall:    {final_rec:.1%}
    F1 Score:  {final_f1:.1%}

Files saved:
  - {model_path}
  - {config_path}

Next steps:
  1. Test calibrated model on all 12 datasets
  2. Compare to original baseline (64.6% AUC)
  3. Decide on further improvements

Run: python3 scripts/step7_evaluate_all_datasets.py
""")
