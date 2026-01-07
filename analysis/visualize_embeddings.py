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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("=" * 80)
print("EMBEDDING VISUALIZATION: Fine-tuned vs Base BERT")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset (sample)...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use small sample for visualization
SAMPLE_SIZE = 500
test_data = dataset['test'].select(range(SAMPLE_SIZE))
print(f"   Using {SAMPLE_SIZE} test pairs")

# Load models
print("\n2. Loading models...")
base_model = SentenceTransformer('bert-base-cased')
finetuned_model = SentenceTransformer('./bert-finetuned-authorship')
print("   Both models loaded!")

# Compute embeddings
print("\n3. Computing embeddings...")
texts = []
labels = []
pair_ids = []

for i, ex in enumerate(test_data):
    texts.append(ex['text1'])
    texts.append(ex['text2'])
    labels.append(1 if ex['same'] else 0)
    labels.append(1 if ex['same'] else 0)
    pair_ids.append(i)
    pair_ids.append(i)

    if (i + 1) % 100 == 0:
        print(f"   {i+1}/{SAMPLE_SIZE}")

print("\n   Computing base BERT embeddings...")
base_embeddings = base_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

print("   Computing fine-tuned BERT embeddings...")
finetuned_embeddings = finetuned_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

print(f"   Embeddings shape: {base_embeddings.shape}")

# Reduce dimensionality with PCA
print("\n4. Reducing dimensionality...")
print("   Applying PCA (768 -> 2 dims)...")
pca = PCA(n_components=2, random_state=42)
base_2d = pca.fit_transform(base_embeddings)

pca2 = PCA(n_components=2, random_state=42)
finetuned_2d = pca2.fit_transform(finetuned_embeddings)
print("   Done!")

# Visualize
print("\n5. Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Base BERT
ax = axes[0]
labels_arr = np.array(labels)
scatter = ax.scatter(base_2d[:, 0], base_2d[:, 1],
                     c=labels_arr, cmap='coolwarm', alpha=0.6, s=20)
ax.set_title('Base BERT Embeddings', fontsize=14, fontweight='bold')
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
plt.colorbar(scatter, ax=ax, label='Same Author (1) vs Different (0)')

# Fine-tuned BERT
ax = axes[1]
scatter = ax.scatter(finetuned_2d[:, 0], finetuned_2d[:, 1],
                     c=labels_arr, cmap='coolwarm', alpha=0.6, s=20)
ax.set_title('Fine-tuned BERT Embeddings', fontsize=14, fontweight='bold')
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
plt.colorbar(scatter, ax=ax, label='Same Author (1) vs Different (0)')

plt.tight_layout()
plt.savefig('bert_embeddings_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: bert_embeddings_comparison.png")

# Compute within-pair distances
print("\n6. Analyzing within-pair distances...")

base_distances = []
finetuned_distances = []
same_author = []

for i in range(0, len(texts), 2):
    # Distance between text1 and text2 in each pair
    base_dist = np.linalg.norm(base_embeddings[i] - base_embeddings[i+1])
    finetuned_dist = np.linalg.norm(finetuned_embeddings[i] - finetuned_embeddings[i+1])

    base_distances.append(base_dist)
    finetuned_distances.append(finetuned_dist)
    same_author.append(labels[i])

base_distances = np.array(base_distances)
finetuned_distances = np.array(finetuned_distances)
same_author = np.array(same_author)

# Plot distance distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Base BERT distances
ax = axes[0]
ax.hist(base_distances[same_author == 1], bins=30, alpha=0.7, label='Same Author', color='blue')
ax.hist(base_distances[same_author == 0], bins=30, alpha=0.7, label='Different Author', color='red')
ax.set_xlabel('Euclidean Distance')
ax.set_ylabel('Count')
ax.set_title('Base BERT: Within-Pair Distances', fontweight='bold')
ax.legend()

# Fine-tuned BERT distances
ax = axes[1]
ax.hist(finetuned_distances[same_author == 1], bins=30, alpha=0.7, label='Same Author', color='blue')
ax.hist(finetuned_distances[same_author == 0], bins=30, alpha=0.7, label='Different Author', color='red')
ax.set_xlabel('Euclidean Distance')
ax.set_ylabel('Count')
ax.set_title('Fine-tuned BERT: Within-Pair Distances', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('bert_distance_distributions.png', dpi=150, bbox_inches='tight')
print("   Saved: bert_distance_distributions.png")

# Statistics
print("\n" + "=" * 80)
print("DISTANCE STATISTICS")
print("=" * 80)

print("\nBase BERT:")
print(f"  Same author pairs:      {base_distances[same_author == 1].mean():.3f} ± {base_distances[same_author == 1].std():.3f}")
print(f"  Different author pairs: {base_distances[same_author == 0].mean():.3f} ± {base_distances[same_author == 0].std():.3f}")
print(f"  Separation: {base_distances[same_author == 0].mean() - base_distances[same_author == 1].mean():.3f}")

print("\nFine-tuned BERT:")
print(f"  Same author pairs:      {finetuned_distances[same_author == 1].mean():.3f} ± {finetuned_distances[same_author == 1].std():.3f}")
print(f"  Different author pairs: {finetuned_distances[same_author == 0].mean():.3f} ± {finetuned_distances[same_author == 0].std():.3f}")
print(f"  Separation: {finetuned_distances[same_author == 0].mean() - finetuned_distances[same_author == 1].mean():.3f}")

improvement = (finetuned_distances[same_author == 0].mean() - finetuned_distances[same_author == 1].mean()) - \
              (base_distances[same_author == 0].mean() - base_distances[same_author == 1].mean())

print(f"\nImprovement in separation: {improvement:.3f} ({improvement/base_distances.mean()*100:.1f}% relative)")

print("\nDone!")
