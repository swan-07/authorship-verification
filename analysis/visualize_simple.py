import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("VISUALIZING BERT EXAMPLES (SIMPLE)")
print("=" * 80)

# Load
print("\nLoading...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

model = SentenceTransformer('bert-base-cased')
test = dataset['test']

# Get 20 examples
print("\nProcessing 20 examples...")
n_samples = 20
labels, texts_1, texts_2, scores = [], [], [], []

for i in range(n_samples):
    ex = test[i]
    e1 = model.encode(ex['text1'])
    e2 = model.encode(ex['text2'])

    cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    labels.append(ex['same'])
    texts_1.append(ex['text1'][:150])
    texts_2.append(ex['text2'][:150])
    scores.append(cos_sim)

labels = np.array(labels)
scores = np.array(scores)

# Create visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Score distribution
ax = axes[0, 0]
ax.hist(scores[labels == 0], bins=10, alpha=0.6, label='Different Authors', color='red')
ax.hist(scores[labels == 1], bins=10, alpha=0.6, label='Same Author', color='blue')
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold')
ax.set_xlabel('Cosine Similarity', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Score Distribution (20 samples)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 2. Scatter plot
ax = axes[0, 1]
ax.scatter(range(len(scores)), scores, c=['blue' if l == 1 else 'red' for l in labels],
           s=100, alpha=0.7, edgecolors='black')
ax.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold')
ax.set_xlabel('Example Index', fontsize=12)
ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Scores per Example', fontsize=14, fontweight='bold')
ax.legend(['Threshold', 'Same Author', 'Different Author'], fontsize=11)
ax.grid(True, alpha=0.3)

# 3 & 4. Show actual examples
same_examples = [(i, scores[i]) for i in range(len(labels)) if labels[i] == 1]
diff_examples = [(i, scores[i]) for i in range(len(labels)) if labels[i] == 0]

for ax_idx, (title, examples, color) in enumerate([
    ("SAME AUTHOR Examples", same_examples[:3], 'lightblue'),
    ("DIFFERENT AUTHORS Examples", diff_examples[:3], 'lightcoral')
]):
    ax = axes[1, ax_idx]
    ax.axis('off')

    text_content = f"{title}\\n{'='*60}\\n\\n"
    for i, (ex_idx, score) in enumerate(examples):
        text_content += f"Example {i+1} (Score: {score:.3f}):\\n"
        text_content += f"Text 1: {texts_1[ex_idx][:100]}...\\n"
        text_content += f"Text 2: {texts_2[ex_idx][:100]}...\\n\\n"

    ax.text(0.05, 0.95, text_content, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    ax.set_title(title, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('bert_examples_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: bert_examples_visualization.png")

# Print to console
print("\n" + "=" * 80)
print("SAMPLE EXAMPLES")
print("=" * 80)

if len(same_examples) > 0:
    print("\nSAME AUTHOR (high score):")
    idx, score = same_examples[0]
    print(f"Score: {score:.3f}")
    print(f"Text 1: {texts_1[idx]}")
    print(f"Text 2: {texts_2[idx]}")

if len(diff_examples) > 0:
    print("\nDIFFERENT AUTHORS (should be low score):")
    idx, score = diff_examples[0]
    print(f"Score: {score:.3f}")
    print(f"Text 1: {texts_1[idx]}")
    print(f"Text 2: {texts_2[idx]}")

print("\nDone!")
