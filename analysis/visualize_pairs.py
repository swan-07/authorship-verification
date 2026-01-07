import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import pickle
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("=" * 80)
print("VISUALIZING TEXT PAIRS (Much Clearer!)")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset (small sample)...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use small sample
SAMPLE_SIZE = 100  # Fewer pairs for clearer visualization
test_data = dataset['test'].select(range(SAMPLE_SIZE))
print(f"   Using {SAMPLE_SIZE} test pairs")

# Load fine-tuned model
print("\n2. Loading fine-tuned model...")
model = SentenceTransformer('./bert-finetuned-authorship')
print("   Model loaded!")

# Compute embeddings for pairs
print("\n3. Computing embeddings and similarities...")
pair_data = []

for i, ex in enumerate(test_data):
    emb1 = model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = model.encode(ex['text2'], convert_to_numpy=True)

    # Cosine similarity
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Euclidean distance
    euclidean_dist = np.linalg.norm(emb1 - emb2)

    pair_data.append({
        'emb1': emb1,
        'emb2': emb2,
        'cos_sim': cos_sim,
        'euclidean': euclidean_dist,
        'same': ex['same']
    })

print(f"   Processed {len(pair_data)} pairs")

# Visualization 1: Lines connecting pairs in 2D space
print("\n4. Creating pair connection visualization...")

# Get all embeddings for PCA
all_embeddings = []
for p in pair_data:
    all_embeddings.append(p['emb1'])
    all_embeddings.append(p['emb2'])
all_embeddings = np.array(all_embeddings)

# PCA to 2D
pca = PCA(n_components=2, random_state=42)
all_2d = pca.fit_transform(all_embeddings)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Show pairs as connected points
ax = axes[0]
for i, p in enumerate(pair_data):
    idx1 = i * 2
    idx2 = i * 2 + 1

    point1 = all_2d[idx1]
    point2 = all_2d[idx2]

    if p['same']:
        color = 'blue'
        alpha = 0.3
        label = 'Same Author' if i == 0 else ''
    else:
        color = 'red'
        alpha = 0.3
        label = 'Different Authors' if i == 0 else ''

    # Draw line connecting the pair
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]],
            color=color, alpha=alpha, linewidth=1)

    # Draw points
    ax.scatter([point1[0]], [point1[1]], c=color, s=20, alpha=0.5)
    ax.scatter([point2[0]], [point2[1]], c=color, s=20, alpha=0.5)

ax.set_xlabel('PCA Dimension 1', fontsize=12)
ax.set_ylabel('PCA Dimension 2', fontsize=12)
ax.set_title('Text Pairs Connected by Lines\n(Blue=Same Author, Red=Different)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='Same Author'),
    Line2D([0], [0], color='red', linewidth=2, label='Different Authors')
]
ax.legend(handles=legend_elements, loc='best')

# Right plot: Distance vs Similarity scatter
ax = axes[1]

same_pairs = [p for p in pair_data if p['same']]
diff_pairs = [p for p in pair_data if not p['same']]

# Plot same author pairs
same_cos = [p['cos_sim'] for p in same_pairs]
same_euc = [p['euclidean'] for p in same_pairs]
ax.scatter(same_cos, same_euc, c='blue', s=50, alpha=0.6, label='Same Author')

# Plot different author pairs
diff_cos = [p['cos_sim'] for p in diff_pairs]
diff_euc = [p['euclidean'] for p in diff_pairs]
ax.scatter(diff_cos, diff_euc, c='red', s=50, alpha=0.6, label='Different Authors')

ax.set_xlabel('Cosine Similarity', fontsize=12)
ax.set_ylabel('Euclidean Distance', fontsize=12)
ax.set_title('Similarity vs Distance for Text Pairs', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pair_connections_visualization.png', dpi=150, bbox_inches='tight')
print("   Saved: pair_connections_visualization.png")

# Visualization 2: Clearer comparison - just the distributions
print("\n5. Creating clearer distribution comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cosine similarity histogram
ax = axes[0, 0]
ax.hist(same_cos, bins=20, alpha=0.7, label='Same Author', color='blue', density=True)
ax.hist(diff_cos, bins=20, alpha=0.7, label='Different Authors', color='red', density=True)
ax.axvline(np.mean(same_cos), color='blue', linestyle='--', linewidth=2)
ax.axvline(np.mean(diff_cos), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Cosine Similarity')
ax.set_ylabel('Density')
ax.set_title('Cosine Similarity Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Euclidean distance histogram
ax = axes[0, 1]
ax.hist(same_euc, bins=20, alpha=0.7, label='Same Author', color='blue', density=True)
ax.hist(diff_euc, bins=20, alpha=0.7, label='Different Authors', color='red', density=True)
ax.axvline(np.mean(same_euc), color='blue', linestyle='--', linewidth=2)
ax.axvline(np.mean(diff_euc), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Euclidean Distance')
ax.set_ylabel('Density')
ax.set_title('Euclidean Distance Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Box plot comparison - Cosine
ax = axes[1, 0]
bp = ax.boxplot([diff_cos, same_cos],
                 labels=['Different', 'Same'],
                 patch_artist=True,
                 showmeans=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('blue')
bp['boxes'][1].set_alpha(0.5)
ax.set_ylabel('Cosine Similarity')
ax.set_title('Cosine Similarity Comparison', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Box plot comparison - Euclidean
ax = axes[1, 1]
bp = ax.boxplot([diff_euc, same_euc],
                 labels=['Different', 'Same'],
                 patch_artist=True,
                 showmeans=True)
bp['boxes'][0].set_facecolor('red')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('blue')
bp['boxes'][1].set_alpha(0.5)
ax.set_ylabel('Euclidean Distance')
ax.set_title('Euclidean Distance Comparison', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('pair_metrics_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: pair_metrics_comparison.png")

# Print statistics
print("\n" + "=" * 80)
print("STATISTICS FOR SAMPLE")
print("=" * 80)

print(f"\nTotal pairs: {len(pair_data)}")
print(f"Same author: {len(same_pairs)}")
print(f"Different authors: {len(diff_pairs)}")

print("\nCosine Similarity:")
print(f"  Same author:      {np.mean(same_cos):.3f} ± {np.std(same_cos):.3f}")
print(f"  Different author: {np.mean(diff_cos):.3f} ± {np.std(diff_cos):.3f}")
print(f"  Separation:       {np.mean(same_cos) - np.mean(diff_cos):.3f}")

print("\nEuclidean Distance:")
print(f"  Same author:      {np.mean(same_euc):.3f} ± {np.std(same_euc):.3f}")
print(f"  Different author: {np.mean(diff_euc):.3f} ± {np.std(diff_euc):.3f}")
print(f"  Separation:       {np.mean(diff_euc) - np.mean(same_euc):.3f}")

print("\nDone!")
