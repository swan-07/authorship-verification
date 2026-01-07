import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("=" * 80)
print("VISUALIZING BERT EXAMPLES")
print("=" * 80)

# Load cached dataset
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

model = SentenceTransformer('bert-base-cased')
test = dataset['test']

# Get 20 examples
print("\nGetting 20 examples...")
n_samples = 20
embeddings_1, embeddings_2, labels, texts_1, texts_2, scores = [], [], [], [], [], []

for i in range(n_samples):
    ex = test[i]
    e1 = model.encode(ex['text1'])
    e2 = model.encode(ex['text2'])

    embeddings_1.append(e1)
    embeddings_2.append(e2)
    labels.append(ex['same'])
    texts_1.append(ex['text1'][:100])  # First 100 chars
    texts_2.append(ex['text2'][:100])

    # Cosine similarity
    cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    scores.append(cos_sim)

    if i % 5 == 0:
        print(f"  {i}/{n_samples}")

embeddings_1 = np.array(embeddings_1)
embeddings_2 = np.array(embeddings_2)
labels = np.array(labels)
scores = np.array(scores)

# Create visualizations
print("\nCreating visualizations...")
fig = plt.figure(figsize=(18, 12))

# 1. t-SNE visualization of all embeddings
print("  Computing t-SNE...")
all_embeddings = np.vstack([embeddings_1, embeddings_2])
all_labels = np.concatenate([labels, labels])  # Labels for each embedding
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(all_embeddings)

ax1 = plt.subplot(2, 3, 1)
same_mask = all_labels == 1
diff_mask = all_labels == 0
ax1.scatter(tsne_results[same_mask, 0], tsne_results[same_mask, 1],
           c='blue', alpha=0.5, s=30, label='Same Author')
ax1.scatter(tsne_results[diff_mask, 0], tsne_results[diff_mask, 1],
           c='red', alpha=0.5, s=30, label='Different Authors')
ax1.set_title('t-SNE: All Embeddings')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Score distribution
ax2 = plt.subplot(2, 3, 2)
ax2.hist(scores[labels == 0], bins=20, alpha=0.5, label='Different Authors')
ax2.hist(scores[labels == 1], bins=20, alpha=0.5, label='Same Author')
ax2.axvline(0.5, color='r', linestyle='--', label='Default Threshold')
ax2.set_xlabel('Cosine Similarity')
ax2.set_ylabel('Count')
ax2.set_title('Score Distribution (20 samples)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Confusion examples
ax3 = plt.subplot(2, 3, 3)
# High confidence correct (same author, high score)
high_conf_correct = np.where((labels == 1) & (scores > 0.9))[0]
# High confidence wrong (different author, high score)
high_conf_wrong = np.where((labels == 0) & (scores > 0.9))[0]
# Low confidence correct (different author, low score)
low_conf_correct = np.where((labels == 0) & (scores < 0.7))[0]

categories = ['High Conf\nCorrect', 'High Conf\nWrong', 'Low Conf\nCorrect']
counts = [len(high_conf_correct), len(high_conf_wrong), len(low_conf_correct)]
colors = ['green', 'red', 'orange']
ax3.bar(categories, counts, color=colors, alpha=0.7)
ax3.set_ylabel('Count')
ax3.set_title('Prediction Categories')
ax3.grid(True, alpha=0.3, axis='y')

# 4-6. Example text pairs
examples_data = [
    ("High Confidence CORRECT", high_conf_correct, 'green'),
    ("High Confidence WRONG", high_conf_wrong, 'red'),
    ("Low Confidence", low_conf_correct, 'orange')
]

for idx, (title, indices, color) in enumerate(examples_data):
    ax = plt.subplot(2, 3, 4 + idx)
    ax.axis('off')

    if len(indices) > 0:
        ex_idx = indices[0]
        text = f"{title}\\n"
        text += f"Score: {scores[ex_idx]:.3f}\\n"
        text += f"Label: {'Same' if labels[ex_idx] == 1 else 'Different'}\\n\\n"
        text += f"Text 1: {texts_1[ex_idx][:80]}...\\n\\n"
        text += f"Text 2: {texts_2[ex_idx][:80]}..."

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.1),
               family='monospace', wrap=True)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('bert_examples_visualization.png', dpi=150, bbox_inches='tight')
print("  Saved: bert_examples_visualization.png")

# Print interesting examples
print("\n" + "=" * 80)
print("INTERESTING EXAMPLES")
print("=" * 80)

if len(high_conf_wrong) > 0:
    print("\nHIGH CONFIDENCE WRONG (Different authors, but BERT says same):")
    for i in high_conf_wrong[:2]:
        print(f"\nScore: {scores[i]:.3f} | Label: Different Authors")
        print(f"Text 1: {texts_1[i]}")
        print(f"Text 2: {texts_2[i]}")

if len(high_conf_correct) > 0:
    print("\nHIGH CONFIDENCE CORRECT (Same author, BERT agrees):")
    for i in high_conf_correct[:2]:
        print(f"\nScore: {scores[i]:.3f} | Label: Same Author")
        print(f"Text 1: {texts_1[i]}")
        print(f"Text 2: {texts_2[i]}")

print("\nDone!")
