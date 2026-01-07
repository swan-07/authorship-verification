import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("Loading test results...")
results = np.load('finetuned_test_results_full.npz')
scores = results['scores']  # Cosine similarity scores
labels = results['labels']

print(f"Total pairs: {len(labels):,}")
print(f"Same author: {labels.sum():,}")
print(f"Different author: {(1-labels).sum():,}")

# Separate scores by label
same_scores = scores[labels == 1]
diff_scores = scores[labels == 0]

print("\nCosine Similarity Statistics:")
print(f"Same author pairs:      {same_scores.mean():.3f} ± {same_scores.std():.3f}")
print(f"Different author pairs: {diff_scores.mean():.3f} ± {diff_scores.std():.3f}")
print(f"Separation:             {same_scores.mean() - diff_scores.mean():.3f}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(same_scores, bins=50, alpha=0.7, label='Same Author', color='blue', density=True)
ax.hist(diff_scores, bins=50, alpha=0.7, label='Different Author', color='red', density=True)
ax.axvline(same_scores.mean(), color='blue', linestyle='--', linewidth=2, label=f'Same Mean: {same_scores.mean():.3f}')
ax.axvline(diff_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Diff Mean: {diff_scores.mean():.3f}')
ax.set_xlabel('Cosine Similarity', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Fine-tuned BERT: Cosine Similarity Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Box plot
ax = axes[1]
bp = ax.boxplot([diff_scores, same_scores],
                 labels=['Different Author', 'Same Author'],
                 patch_artist=True,
                 showmeans=True,
                 meanprops=dict(marker='D', markerfacecolor='yellow', markersize=8))

# Color the boxes
bp['boxes'][0].set_facecolor('red')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('blue')
bp['boxes'][1].set_alpha(0.5)

ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Fine-tuned BERT: Similarity by Author Match', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add statistics as text
stats_text = f'Same Author: μ={same_scores.mean():.3f}, σ={same_scores.std():.3f}\n'
stats_text += f'Different Author: μ={diff_scores.mean():.3f}, σ={diff_scores.std():.3f}\n'
stats_text += f'Separation: {same_scores.mean() - diff_scores.mean():.3f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('cosine_similarity_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved: cosine_similarity_distribution.png")

# Create a second figure comparing overlap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Violin plot for better overlap visualization
parts = ax.violinplot([diff_scores, same_scores],
                       positions=[1, 2],
                       showmeans=True,
                       showmedians=True,
                       widths=0.7)

# Color the violins
for i, pc in enumerate(parts['bodies']):
    if i == 0:
        pc.set_facecolor('red')
        pc.set_alpha(0.6)
    else:
        pc.set_facecolor('blue')
        pc.set_alpha(0.6)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Different Author', 'Same Author'], fontsize=12)
ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Fine-tuned BERT: Distribution Overlap Analysis', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add percentile lines
for percentile, linestyle in [(25, ':'), (50, '-'), (75, ':')]:
    same_p = np.percentile(same_scores, percentile)
    diff_p = np.percentile(diff_scores, percentile)

plt.tight_layout()
plt.savefig('cosine_similarity_violin.png', dpi=150, bbox_inches='tight')
print("Saved: cosine_similarity_violin.png")

# Print detailed statistics
print("\n" + "=" * 60)
print("DETAILED STATISTICS")
print("=" * 60)

print("\nSame Author Pairs:")
print(f"  Count:  {len(same_scores):,}")
print(f"  Mean:   {same_scores.mean():.4f}")
print(f"  Std:    {same_scores.std():.4f}")
print(f"  Min:    {same_scores.min():.4f}")
print(f"  25th:   {np.percentile(same_scores, 25):.4f}")
print(f"  Median: {np.median(same_scores):.4f}")
print(f"  75th:   {np.percentile(same_scores, 75):.4f}")
print(f"  Max:    {same_scores.max():.4f}")

print("\nDifferent Author Pairs:")
print(f"  Count:  {len(diff_scores):,}")
print(f"  Mean:   {diff_scores.mean():.4f}")
print(f"  Std:    {diff_scores.std():.4f}")
print(f"  Min:    {diff_scores.min():.4f}")
print(f"  25th:   {np.percentile(diff_scores, 25):.4f}")
print(f"  Median: {np.median(diff_scores):.4f}")
print(f"  75th:   {np.percentile(diff_scores, 75):.4f}")
print(f"  Max:    {diff_scores.max():.4f}")

print("\nOverlap Analysis:")
overlap_start = max(diff_scores.min(), same_scores.min())
overlap_end = min(diff_scores.max(), same_scores.max())
print(f"  Overlap range: [{overlap_start:.4f}, {overlap_end:.4f}]")

# Calculate percentage in overlap region
same_in_overlap = ((same_scores >= overlap_start) & (same_scores <= overlap_end)).sum()
diff_in_overlap = ((diff_scores >= overlap_start) & (diff_scores <= overlap_end)).sum()
print(f"  Same author pairs in overlap: {same_in_overlap:,} ({same_in_overlap/len(same_scores)*100:.1f}%)")
print(f"  Different author pairs in overlap: {diff_in_overlap:,} ({diff_in_overlap/len(diff_scores)*100:.1f}%)")

print("\nDone!")
