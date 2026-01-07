import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

print("Creating improved calibration visualization...")

# Load calibration model
with open('calibration_model_base_bert.pkl', 'rb') as f:
    cal_model = pickle.load(f)
with open('calibration_config.pkl', 'rb') as f:
    config = pickle.load(f)

# Load test scores
data = np.load('test_scores_cached.npz')
raw_scores = data['scores'].reshape(-1, 1)
labels = data['labels']

# Get calibrated probabilities
cal_probs = cal_model.predict_proba(raw_scores)[:, 1]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(labels, cal_probs)
auc = roc_auc_score(labels, cal_probs)
axes[0, 0].plot(fpr, tpr, label=f'Calibrated BERT (AUC={auc:.3f})', linewidth=2.5, color='blue')
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. RAW Cosine Similarity Distribution
axes[0, 1].hist(raw_scores[labels == 0], bins=50, alpha=0.6, label='Different Authors', color='red', density=True)
axes[0, 1].hist(raw_scores[labels == 1], bins=50, alpha=0.6, label='Same Author', color='blue', density=True)
axes[0, 1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
axes[0, 1].set_xlabel('Raw Cosine Similarity', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title('BEFORE Calibration: Raw Scores', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(0.5, 0.95, 'Problem: Almost all scores > 0.76\nThreshold 0.5 predicts "same" for everything',
               transform=axes[0, 1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# 3. CALIBRATED Probability Distribution
axes[1, 0].hist(cal_probs[labels == 0], bins=50, alpha=0.6, label='Different Authors', color='red', density=True)
axes[1, 0].hist(cal_probs[labels == 1], bins=50, alpha=0.6, label='Same Author', color='blue', density=True)
axes[1, 0].axvline(config['best_threshold'], color='green', linestyle='--', linewidth=2.5,
                  label=f'Optimal Threshold ({config["best_threshold"]:.3f})')
axes[1, 0].set_xlabel('Calibrated Probability', fontsize=12)
axes[1, 0].set_ylabel('Density', fontsize=12)
axes[1, 0].set_title('AFTER Calibration: Calibrated Probabilities', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(0.5, 0.95, 'Fixed: Probabilities spread across 0-1\nThreshold separates classes better',
               transform=axes[1, 0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# 4. Scatter: Raw vs Calibrated
sample_indices = np.random.choice(len(raw_scores), 1000, replace=False)
axes[1, 1].scatter(raw_scores[sample_indices], cal_probs[sample_indices],
                  c=['blue' if l == 1 else 'red' for l in labels[sample_indices]],
                  alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x (no change)')
axes[1, 1].axhline(config['best_threshold'], color='green', linestyle='--', linewidth=2,
                  label=f'Optimal Threshold ({config["best_threshold"]:.3f})')
axes[1, 1].set_xlabel('Raw Cosine Similarity', fontsize=12)
axes[1, 1].set_ylabel('Calibrated Probability', fontsize=12)
axes[1, 1].set_title('Calibration Mapping (1000 samples)', fontsize=14, fontweight='bold')
axes[1, 1].legend(['Same Author', 'Different Authors', 'y=x', 'Threshold'], fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: calibration_comparison.png")

# Print statistics
print("\n" + "=" * 80)
print("SCORE STATISTICS")
print("=" * 80)
print("\nRaw Cosine Similarity:")
print(f"  Same Author:      mean={raw_scores[labels==1].mean():.3f}, std={raw_scores[labels==1].std():.3f}")
print(f"  Different Author: mean={raw_scores[labels==0].mean():.3f}, std={raw_scores[labels==0].std():.3f}")
print(f"  Overlap: {((raw_scores[labels==1].min() < raw_scores[labels==0].max()))}")

print("\nCalibrated Probabilities:")
print(f"  Same Author:      mean={cal_probs[labels==1].mean():.3f}, std={cal_probs[labels==1].std():.3f}")
print(f"  Different Author: mean={cal_probs[labels==0].mean():.3f}, std={cal_probs[labels==0].std():.3f}")
print(f"  Better separation: {(cal_probs[labels==1].mean() - cal_probs[labels==0].mean()):.3f}")

print("\nDone!")
