"""
View Base BERT Results
"""

import numpy as np

print("="*80)
print("BASE BERT RESULTS")
print("="*80)

# Load results
results = np.load('/Users/swan/Documents/GitHub/authorship-verification/results_base_bert.npz')

scores = results['scores']
labels = results['labels']
predictions = results['predictions']

print(f"\nLoaded results:")
print(f"  Total samples: {len(scores)}")
print(f"  Same author (label=1): {sum(labels == 1)}")
print(f"  Different author (label=0): {sum(labels == 0)}")

# Accuracy with threshold 0.5
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

print(f"\nPerformance (threshold=0.5):")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1 Score:  {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(labels, predictions)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Score distribution
same_scores = scores[labels == 1]
diff_scores = scores[labels == 0]

print(f"\nScore Distribution:")
print(f"  Same author (n={len(same_scores)}):")
print(f"    Mean:   {np.mean(same_scores):.3f}")
print(f"    Median: {np.median(same_scores):.3f}")
print(f"    Std:    {np.std(same_scores):.3f}")
print(f"    Range:  {np.min(same_scores):.3f} - {np.max(same_scores):.3f}")

print(f"\n  Different author (n={len(diff_scores)}):")
print(f"    Mean:   {np.mean(diff_scores):.3f}")
print(f"    Median: {np.median(diff_scores):.3f}")
print(f"    Std:    {np.std(diff_scores):.3f}")
print(f"    Range:  {np.min(diff_scores):.3f} - {np.max(diff_scores):.3f}")

separation = abs(np.mean(same_scores) - np.mean(diff_scores))
print(f"\n  Separation: {separation:.3f}")

# Find best threshold
print(f"\nFinding optimal threshold...")
best_f1 = 0
best_thresh = 0.5

for thresh in np.arange(0.05, 0.95, 0.01):
    preds = (scores >= thresh).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"  Best threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

# Evaluate at best threshold
best_preds = (scores >= best_thresh).astype(int)
best_acc = accuracy_score(labels, best_preds)
best_prec, best_rec, _, _ = precision_recall_fscore_support(labels, best_preds, average='binary')

print(f"\nPerformance at best threshold ({best_thresh:.2f}):")
print(f"  Accuracy:  {best_acc:.3f}")
print(f"  Precision: {best_prec:.3f}")
print(f"  Recall:    {best_rec:.3f}")
print(f"  F1 Score:  {best_f1:.3f}")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if separation < 0.01:
    print("\n✗ FAILED - No separation (model broken)")
elif separation < 0.05:
    print("\n⚠ POOR - Very weak separation")
elif separation < 0.10:
    print("\n~ OKAY - Some separation, needs improvement")
elif separation < 0.20:
    print("\n✓ GOOD - Reasonable separation")
else:
    print("\n✓✓ EXCELLENT - Strong separation")

print(f"\nBase BERT vs Fine-tuned BERT:")
print(f"  Fine-tuned: Separation = 0.000 (BROKEN)")
print(f"  Base BERT:  Separation = {separation:.3f}")

if separation > 0.01:
    print(f"\n✓ Base BERT is WORKING and better than fine-tuned!")
    print(f"\nNext steps:")
    print(f"  1. Train calibration model")
    print(f"  2. Test on all 12 datasets")
    print(f"  3. Compare to original baselines")
else:
    print(f"\n✗ Base BERT also has issues")
    print(f"  May need to try different model or approach")
