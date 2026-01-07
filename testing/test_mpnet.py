import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("TESTING ALL-MPNET-BASE-V2 (Better Sentence Transformer)")
print("=" * 80)

# Load cached dataset
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Load better model
print("\nLoading all-mpnet-base-v2 model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("✓ Model loaded")

# 1. Compute scores on VALIDATION
print("\n1. Computing scores on VALIDATION...")
val = dataset['validation']
val_scores, val_labels = [], []
for i, ex in enumerate(val):
    if i % 1000 == 0: print(f"  {i}/{len(val)}")
    e1, e2 = model.encode(ex['text1']), model.encode(ex['text2'])
    val_scores.append(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    val_labels.append(ex['same'])

val_scores = np.array(val_scores).reshape(-1, 1)
val_labels = np.array(val_labels)

# 2. Train calibration on VALIDATION
print("\n2. Training calibration on VALIDATION...")
cal_model = LogisticRegression()
cal_model.fit(val_scores, val_labels)

# 3. Find optimal threshold
print("\n3. Finding optimal threshold...")
val_probs = cal_model.predict_proba(val_scores)[:, 1]
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
best_idx = np.argmax(tpr - fpr)
optimal_thresh = thresholds[best_idx]
print(f"  Optimal threshold: {optimal_thresh:.4f}")

# 4. Compute scores on TEST
print("\n4. Computing scores on TEST...")
test = dataset['test']
test_scores, test_labels = [], []
for i, ex in enumerate(test):
    if i % 1000 == 0: print(f"  {i}/{len(test)}")
    e1, e2 = model.encode(ex['text1']), model.encode(ex['text2'])
    test_scores.append(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    test_labels.append(ex['same'])

test_scores = np.array(test_scores).reshape(-1, 1)
test_labels = np.array(test_labels)

# 5. Get calibrated probabilities
print("\n5. Computing calibrated probabilities...")
test_probs = cal_model.predict_proba(test_scores)[:, 1]
preds = (test_probs >= optimal_thresh).astype(int)

# Results
print("\n" + "=" * 80)
print("RESULTS: ALL-MPNET-BASE-V2 + CALIBRATION")
print("=" * 80)
print(f"Threshold: {optimal_thresh:.4f}")
print(f"Accuracy:  {accuracy_score(test_labels, preds):.1%}")
print(f"Precision: {precision_score(test_labels, preds):.1%}")
print(f"Recall:    {recall_score(test_labels, preds):.1%}")
print(f"F1 Score:  {f1_score(test_labels, preds):.1%}")
auc = roc_auc_score(test_labels, test_probs)
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Base BERT + Calibration:       Acc=63.7%, Prec=68.6%, Rec=50.5%, F1=58.2%")
print(f"MPNet + Calibration:           Acc={accuracy_score(test_labels, preds):.1%}, Prec={precision_score(test_labels, preds):.1%}, Rec={recall_score(test_labels, preds):.1%}, F1={f1_score(test_labels, preds):.1%}")

improvement = (accuracy_score(test_labels, preds) - 0.637) * 100
print(f"\nImprovement: {improvement:+.1f} percentage points")

# Save model
print("\n6. Saving model...")
with open('mpnet_calibration_model.pkl', 'wb') as f:
    pickle.dump(cal_model, f)
with open('mpnet_calibration_config.pkl', 'wb') as f:
    pickle.dump({'best_threshold': float(optimal_thresh)}, f)
np.savez('mpnet_test_scores.npz', scores=test_scores, labels=test_labels, probs=test_probs)
print("  Saved models and scores")

# Create visualization
print("\n7. Creating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve comparison
fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs)
axes[0].plot(fpr_test, tpr_test, label=f'MPNet (AUC={auc:.3f})', linewidth=2.5, color='green')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve: MPNet')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Score distribution
axes[1].hist(test_scores[test_labels == 0], bins=50, alpha=0.6, label='Different', color='red', density=True)
axes[1].hist(test_scores[test_labels == 1], bins=50, alpha=0.6, label='Same', color='blue', density=True)
axes[1].set_xlabel('Cosine Similarity')
axes[1].set_ylabel('Density')
axes[1].set_title('MPNet Score Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mpnet_results.png', dpi=150)
print("  Saved: mpnet_results.png")

print("\nDone!")
