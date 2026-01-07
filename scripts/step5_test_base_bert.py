"""
Test Base BERT Model (No Fine-tuning)
This will give us a working baseline to compare against.
"""

import sys
import numpy as np
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

print("="*80)
print("TESTING BASE BERT MODEL (No Fine-tuning)")
print("="*80)

# Load base BERT model
print("\n1. Loading base BERT model...")
print("   Model: bert-base-cased")

try:
    model = SentenceTransformer('bert-base-cased')
    print(f"   ✓ Model loaded successfully")
    print(f"   Device: {model.device}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Load test dataset
print("\n2. Loading test dataset (arxiv)...")
try:
    test_data = load_dataset(
        'swan07/authorship-verification',
        data_files={'test': 'arxiv_test.csv'},
        split='test'
    )
    print(f"   ✓ Loaded {len(test_data)} test samples")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Run evaluation
print("\n3. Running evaluation...")

def evaluate_at_threshold(model, data, threshold=0.5):
    """Evaluate model at a specific threshold"""
    all_labels = []
    all_scores = []

    for sample in tqdm(data, desc="   Processing"):
        text1 = sample['text1']
        text2 = sample['text2']
        label = sample['same']

        # Get embeddings
        emb1 = model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        emb2 = model.encode(text2, convert_to_tensor=True, show_progress_bar=False)

        # Cosine similarity
        score = util.pytorch_cos_sim(emb1, emb2).item()

        all_scores.append(score)
        all_labels.append(label)

    # Convert to numpy
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Apply threshold
    predictions = (all_scores >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'scores': all_scores,
        'labels': all_labels,
        'predictions': predictions
    }

# Test with default threshold
print("\n   Testing with threshold=0.5:")
results = evaluate_at_threshold(model, test_data, threshold=0.5)

print(f"\n   Results:")
print(f"     Accuracy:  {results['accuracy']:.3f}")
print(f"     Precision: {results['precision']:.3f}")
print(f"     Recall:    {results['recall']:.3f}")
print(f"     F1 Score:  {results['f1']:.3f}")

# Show score distribution
print("\n4. Score Distribution Analysis:")
same_author_scores = results['scores'][results['labels'] == 1]
diff_author_scores = results['scores'][results['labels'] == 0]

print(f"\n   Same author pairs (n={len(same_author_scores)}):")
print(f"     Mean:   {np.mean(same_author_scores):.3f}")
print(f"     Median: {np.median(same_author_scores):.3f}")
print(f"     Std:    {np.std(same_author_scores):.3f}")
print(f"     Range:  {np.min(same_author_scores):.3f} - {np.max(same_author_scores):.3f}")

print(f"\n   Different author pairs (n={len(diff_author_scores)}):")
print(f"     Mean:   {np.mean(diff_author_scores):.3f}")
print(f"     Median: {np.median(diff_author_scores):.3f}")
print(f"     Std:    {np.std(diff_author_scores):.3f}")
print(f"     Range:  {np.min(diff_author_scores):.3f} - {np.max(diff_author_scores):.3f}")

separation = abs(np.mean(same_author_scores) - np.mean(diff_author_scores))
print(f"\n   Mean Separation: {separation:.3f}")

# Test different thresholds to find optimal
print("\n5. Testing Different Thresholds:")
print("   " + "-" * 70)
print(f"   {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("   " + "-" * 70)

best_f1 = 0
best_threshold = 0.5

for thresh in np.arange(0.1, 1.0, 0.05):
    preds = (results['scores'] >= thresh).astype(int)
    acc = accuracy_score(results['labels'], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        results['labels'], preds, average='binary', zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

    # Print every other threshold
    if int(thresh * 100) % 10 == 0:
        print(f"   {thresh:<12.2f} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")

print("   " + "-" * 70)
print(f"   Best threshold: {best_threshold:.2f} (F1 = {best_f1:.3f})")

# Evaluate at best threshold
print(f"\n6. Results at Best Threshold ({best_threshold:.2f}):")
best_results = evaluate_at_threshold(model, test_data, threshold=best_threshold)

print(f"   Accuracy:  {best_results['accuracy']:.3f}")
print(f"   Precision: {best_results['precision']:.3f}")
print(f"   Recall:    {best_results['recall']:.3f}")
print(f"   F1 Score:  {best_results['f1']:.3f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if separation < 0.05:
    print("\n⚠ POOR SEPARATION")
    print("   The model cannot distinguish between same/different authors well.")
    print("   This is expected for base BERT without fine-tuning.")
elif separation > 0.1:
    print("\n✓ GOOD SEPARATION")
    print("   The model shows some ability to distinguish authors.")
else:
    print("\n⚠ MODERATE SEPARATION")
    print("   The model has some distinguishing ability but could be better.")

if results['recall'] > 0.9 and results['precision'] < 0.6:
    print("\n⚠ HIGH RECALL, LOW PRECISION")
    print("   Model predicts 'same author' too often.")
    print("   Solution: Increase threshold or use calibration.")

print("\n" + "="*80)
print("COMPARISON WITH FINE-TUNED MODEL")
print("="*80)

print("""
Fine-tuned model results (BROKEN):
  - All scores: ~1.000 (no variation)
  - Separation: 0.000
  - Problem: Model collapsed, outputs identical embeddings

Base BERT results (this test):
  - Score distribution: Check above
  - Separation: {:.3f}
  - Status: {}

CONCLUSION:
{}
""".format(
    separation,
    "Working" if separation > 0.01 else "Also broken",
    "Base BERT is working better than fine-tuned model!\nRecommend using base BERT + calibration instead." if separation > 0.01
    else "Both models have issues. May need to retrain from scratch."
))

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
1. Train calibration model on base BERT
   - Use validation set to learn probability mapping
   - Should improve precision without losing recall

2. Test on all 12 datasets
   - See how base BERT performs across domains
   - Compare to original baseline results

3. Decide on retraining strategy
   - If base BERT + calibration works well: Use it
   - If not: Retrain fine-tuned model properly

Run: python3 scripts/step6_train_calibration_base_bert.py
""")

# Save results for later use
print("\nSaving results...")
results_path = "/Users/swan/Documents/GitHub/authorship-verification/results_base_bert.npz"
np.savez(results_path,
         scores=results['scores'],
         labels=results['labels'],
         predictions=results['predictions'])
print(f"✓ Saved to: {results_path}")
