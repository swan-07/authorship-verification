"""
Phase 1, Step 4: Test BERT Model
Downloads BERT model from HuggingFace and runs basic evaluation.
"""

import sys
from pathlib import Path

print("="*80)
print("STEP 4: TESTING BERT MODEL")
print("="*80)

# Download BERT model from HuggingFace
print("\n1. Downloading BERT model from HuggingFace...")
print("   Repository: swan07/final-models")

try:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer, util
    import torch
    import numpy as np
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from tqdm import tqdm

    # Download the model
    print("   Downloading... (this may take a few minutes)")
    model_path = snapshot_download(
        repo_id="swan07/final-models",
        repo_type="dataset",
        allow_patterns="bertmodel/*"
    )

    # The model is in a subdirectory
    bertmodel_path = Path(model_path) / "bertmodel"
    print(f"   ✓ Downloaded to: {bertmodel_path}")

    # Load the model
    print("\n2. Loading BERT model...")
    model = SentenceTransformer(str(bertmodel_path))
    print(f"   ✓ Model loaded successfully")
    print(f"   Device: {model.device}")

except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    print("\n   Falling back to base BERT model...")
    try:
        model = SentenceTransformer('bert-base-cased')
        print("   ✓ Using bert-base-cased (not fine-tuned)")
    except Exception as e2:
        print(f"   ✗ Error: {e2}")
        sys.exit(1)

# Load a small test dataset
print("\n3. Loading test dataset (arxiv)...")
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

# Evaluate with different thresholds
print("\n4. Running evaluation WITHOUT calibration...")
print("   (Testing different thresholds)")

def evaluate_at_threshold(model, data, threshold=0.5, max_samples=None):
    """Evaluate model at a specific threshold"""
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    all_labels = []
    all_scores = []

    print(f"   Evaluating {len(data)} samples...")
    for i, sample in enumerate(tqdm(data, desc="   Processing")):
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
        'labels': all_labels
    }

# Test with default threshold
print("\n   Testing with threshold=0.5 (default):")
results_05 = evaluate_at_threshold(model, test_data, threshold=0.5, max_samples=106)

print(f"     Accuracy:  {results_05['accuracy']:.3f}")
print(f"     Precision: {results_05['precision']:.3f}")
print(f"     Recall:    {results_05['recall']:.3f}")
print(f"     F1 Score:  {results_05['f1']:.3f}")

# Test with different thresholds
print("\n   Testing different thresholds:")
print("   " + "-" * 60)
print(f"   {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("   " + "-" * 60)

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (results_05['scores'] >= thresh).astype(int)
    acc = accuracy_score(results_05['labels'], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        results_05['labels'], preds, average='binary', zero_division=0
    )
    print(f"   {thresh:<12.1f} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")

print("   " + "-" * 60)

# Show score distribution
print("\n5. Score Distribution Analysis:")
same_author_scores = results_05['scores'][results_05['labels'] == 1]
diff_author_scores = results_05['scores'][results_05['labels'] == 0]

print(f"   Same author pairs (n={len(same_author_scores)}):")
print(f"     Mean:   {np.mean(same_author_scores):.3f}")
print(f"     Median: {np.median(same_author_scores):.3f}")
print(f"     Range:  {np.min(same_author_scores):.3f} - {np.max(same_author_scores):.3f}")

print(f"\n   Different author pairs (n={len(diff_author_scores)}):")
print(f"     Mean:   {np.mean(diff_author_scores):.3f}")
print(f"     Median: {np.median(diff_author_scores):.3f}")
print(f"     Range:  {np.min(diff_author_scores):.3f} - {np.max(diff_author_scores):.3f}")

# Overlap analysis
print(f"\n   Overlap: {np.mean(same_author_scores):.3f} (same) vs {np.mean(diff_author_scores):.3f} (diff)")
print(f"   Separation: {abs(np.mean(same_author_scores) - np.mean(diff_author_scores)):.3f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

if results_05['recall'] > 0.9 and results_05['precision'] < 0.6:
    print("\n⚠ ISSUE DETECTED: High recall, low precision")
    print("   Problem: Model predicts 'same author' too often")
    print("   Cause: Threshold is too low for this domain")
    print("   Solution: Use calibration (logistic regression)")

if abs(np.mean(same_author_scores) - np.mean(diff_author_scores)) < 0.1:
    print("\n⚠ ISSUE DETECTED: Poor score separation")
    print("   Problem: Same/different author scores overlap heavily")
    print("   This means: Model isn't learning strong distinctions")
    print("   Solutions:")
    print("     1. Better training (more epochs, better loss function)")
    print("     2. Data augmentation")
    print("     3. Different base model (RoBERTa, DeBERTa)")

print("\n" + "="*80)
print("✓ EVALUATION COMPLETE")
print("="*80)
print("\nNext steps:")
print("  1. Train calibration model (scripts/step4_train_calibration.py)")
print("  2. Re-evaluate with calibration")
print("  3. Test on all 12 datasets")
