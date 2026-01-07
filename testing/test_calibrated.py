import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

print("=" * 80)
print("TRAIN CALIBRATION ON VAL, TEST ON TEST")
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

val_scores = np.array(val_scores).reshape(-1, 1)
val_labels = np.array(val_labels)

# 2. Train calibration model on VALIDATION
print("\n2. Training calibration on VALIDATION...")
calibration_model = LogisticRegression()
calibration_model.fit(val_scores, val_labels)

# 3. Find optimal threshold on VALIDATION (using calibrated probabilities)
print("\n3. Finding optimal threshold on VALIDATION...")
val_probs = calibration_model.predict_proba(val_scores)[:, 1]
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
optimal_thresh = thresholds[best_idx]
print(f"  Optimal threshold: {optimal_thresh:.4f}")

# Save calibration model
with open('calibration_model_base_bert.pkl', 'wb') as f:
    pickle.dump(calibration_model, f)
with open('calibration_config.pkl', 'wb') as f:
    pickle.dump({'best_threshold': float(optimal_thresh)}, f)
print("  Saved calibration model")

# 4. Load test scores
print("\n4. Loading test scores...")
data = np.load('test_scores_cached.npz')
test_scores = data['scores'].reshape(-1, 1)
test_labels = data['labels']

# 5. Get calibrated probabilities for test
print("\n5. Computing calibrated probabilities on TEST...")
test_probs = calibration_model.predict_proba(test_scores)[:, 1]

# 6. Make predictions
preds = (test_probs >= optimal_thresh).astype(int)

print("\n" + "=" * 80)
print("CALIBRATED BERT RESULTS")
print("=" * 80)
print(f"Threshold: {optimal_thresh:.4f} (from validation)")
print(f"Accuracy:  {accuracy_score(test_labels, preds):.1%}")
print(f"Precision: {precision_score(test_labels, preds):.1%}")
print(f"Recall:    {recall_score(test_labels, preds):.1%}")
print(f"F1 Score:  {f1_score(test_labels, preds):.1%}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Base BERT (threshold=0.5):     Acc=50.0%, Prec=50.0%, Rec=99.9%, F1=66.6%")
print(f"Calibrated BERT:               Acc={accuracy_score(test_labels, preds):.1%}, Prec={precision_score(test_labels, preds):.1%}, Rec={recall_score(test_labels, preds):.1%}, F1={f1_score(test_labels, preds):.1%}")

# 7. Create visualization
print("\n7. Creating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs)
auc = roc_auc_score(test_labels, test_probs)
axes[0].plot(fpr_test, tpr_test, label=f'Calibrated BERT (AUC={auc:.3f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Score distributions
axes[1].hist(test_scores[test_labels == 0], bins=50, alpha=0.5, label='Different Authors', density=True)
axes[1].hist(test_scores[test_labels == 1], bins=50, alpha=0.5, label='Same Author', density=True)
axes[1].axvline(optimal_thresh, color='r', linestyle='--', label=f'Threshold={optimal_thresh:.3f}')
axes[1].set_xlabel('Cosine Similarity')
axes[1].set_ylabel('Density')
axes[1].set_title('Score Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_results.png', dpi=150, bbox_inches='tight')
print("  Saved: calibration_results.png")

print("\nDone!")
