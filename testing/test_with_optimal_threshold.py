import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
import pickle
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("=" * 80)
print("FINDING OPTIMAL THRESHOLD ON VAL, TESTING ON TEST")
print("=" * 80)

# Load cached dataset
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

model = SentenceTransformer('bert-base-cased')

# 1. Get scores on VALIDATION
print("\n1. Computing scores on VALIDATION...")
val = dataset['validation']
val_scores, val_labels = [], []
for i, ex in enumerate(val):
    if i % 1000 == 0: print(f"  {i}/{len(val)}")
    e1, e2 = model.encode(ex['text1']), model.encode(ex['text2'])
    val_scores.append(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    val_labels.append(ex['same'])

val_scores = np.array(val_scores)
val_labels = np.array(val_labels)

# 2. Find optimal threshold on VALIDATION
print("\n2. Finding optimal threshold on VALIDATION...")
fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
optimal_thresh = thresholds[best_idx]
print(f"  Optimal threshold: {optimal_thresh:.4f}")

# 3. Load test scores (already computed)
print("\n3. Loading test scores...")
data = np.load('test_scores_cached.npz')
test_scores = data['scores']
test_labels = data['labels']

# 4. Test with optimal threshold
print("\n" + "=" * 80)
print("BASE BERT WITH OPTIMAL THRESHOLD")
print("=" * 80)
preds = (test_scores >= optimal_thresh).astype(int)
print(f"Threshold: {optimal_thresh:.4f} (from validation)")
print(f"Accuracy:  {accuracy_score(test_labels, preds):.1%}")
print(f"Precision: {precision_score(test_labels, preds):.1%}")
print(f"Recall:    {recall_score(test_labels, preds):.1%}")
print(f"F1 Score:  {f1_score(test_labels, preds):.1%}")
