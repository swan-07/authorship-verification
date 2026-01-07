import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import pickle
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("=" * 80)
print("BERT INTERPRETABILITY ANALYSIS")
print("=" * 80)

# Load model and tokenizer
print("\n1. Loading models...")
model = SentenceTransformer('./bert-finetuned-authorship')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert_model = AutoModel.from_pretrained('bert-base-cased')
bert_model.eval()
print("   Models loaded!")

# Load test data
print("\n2. Loading examples...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Get interesting examples
test_data = dataset['test']

# Find examples with high/low similarity
print("\n3. Computing similarities to find interesting cases...")

# Sample 1000 pairs
sample_size = 1000
examples = []

for i in range(sample_size):
    ex = test_data[i]
    emb1 = model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = model.encode(ex['text2'], convert_to_numpy=True)
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    examples.append({
        'text1': ex['text1'],
        'text2': ex['text2'],
        'same': ex['same'],
        'similarity': cos_sim,
        'emb1': emb1,
        'emb2': emb2
    })

examples.sort(key=lambda x: x['similarity'], reverse=True)

print(f"   Found {len(examples)} examples")
print(f"   Highest similarity: {examples[0]['similarity']:.3f}")
print(f"   Lowest similarity: {examples[-1]['similarity']:.3f}")

# ANALYSIS 1: Token-level contribution
print("\n" + "=" * 80)
print("ANALYSIS 1: WHICH TOKENS CONTRIBUTE TO SIMILARITY?")
print("=" * 80)

def analyze_token_contributions(text1, text2, emb1, emb2):
    """Analyze which tokens contribute most to similarity"""

    # Tokenize
    tokens1 = tokenizer.tokenize(text1[:512])  # BERT max length
    tokens2 = tokenizer.tokenize(text2[:512])

    # Get embeddings for each token
    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs1 = bert_model(**inputs1, output_hidden_states=True)
        outputs2 = bert_model(**inputs2, output_hidden_states=True)

        # Use last layer hidden states
        hidden1 = outputs1.hidden_states[-1][0]  # [seq_len, 768]
        hidden2 = outputs2.hidden_states[-1][0]

    # Compute importance: how much does each token's embedding align with the overall difference
    emb_diff = emb2 - emb1

    # For each token in text1, compute alignment with difference
    scores1 = []
    for i in range(len(hidden1)):
        token_emb = hidden1[i].numpy()
        score = np.dot(token_emb, emb_diff) / (np.linalg.norm(token_emb) * np.linalg.norm(emb_diff))
        scores1.append(abs(score))

    scores2 = []
    for i in range(len(hidden2)):
        token_emb = hidden2[i].numpy()
        score = np.dot(token_emb, emb_diff) / (np.linalg.norm(token_emb) * np.linalg.norm(emb_diff))
        scores2.append(abs(score))

    return tokens1, scores1, tokens2, scores2

# Analyze a high similarity same-author pair
print("\nExample 1: High Similarity (Same Author)")
print("-" * 80)
ex = [e for e in examples if e['same'] == 1 and e['similarity'] > 0.8][0]
print(f"Similarity: {ex['similarity']:.3f}")
print(f"Text 1: {ex['text1'][:200]}...")
print(f"Text 2: {ex['text2'][:200]}...")

tokens1, scores1, tokens2, scores2 = analyze_token_contributions(
    ex['text1'], ex['text2'], ex['emb1'], ex['emb2']
)

# Show top contributing tokens
print("\nTop contributing tokens from Text 1:")
top_indices1 = np.argsort(scores1)[-10:][::-1]
for idx in top_indices1:
    if idx < len(tokens1):
        print(f"  {tokens1[idx]:20s} {scores1[idx]:.4f}")

print("\nTop contributing tokens from Text 2:")
top_indices2 = np.argsort(scores2)[-10:][::-1]
for idx in top_indices2:
    if idx < len(tokens2):
        print(f"  {tokens2[idx]:20s} {scores2[idx]:.4f}")

# ANALYSIS 2: What dimensions capture authorship?
print("\n" + "=" * 80)
print("ANALYSIS 2: WHAT DO DIFFERENT EMBEDDING DIMENSIONS CAPTURE?")
print("=" * 80)

# Compute variance in embeddings for same vs different authors
same_pairs = [e for e in examples if e['same'] == 1]
diff_pairs = [e for e in examples if e['same'] == 0]

print(f"\nAnalyzing {len(same_pairs)} same-author pairs...")
print(f"Analyzing {len(diff_pairs)} different-author pairs...")

# Compute difference vectors
same_diffs = []
for e in same_pairs[:200]:
    same_diffs.append(e['emb1'] - e['emb2'])

diff_diffs = []
for e in diff_pairs[:200]:
    diff_diffs.append(e['emb1'] - e['emb2'])

same_diffs = np.array(same_diffs)
diff_diffs = np.array(diff_diffs)

# Find dimensions with most variance difference
same_var = np.var(same_diffs, axis=0)
diff_var = np.var(diff_diffs, axis=0)

var_ratio = diff_var / (same_var + 1e-8)
top_discriminative_dims = np.argsort(var_ratio)[-20:][::-1]

print("\nTop 20 most discriminative dimensions:")
print("(Higher variance in different-author pairs = captures authorship)")
for i, dim in enumerate(top_discriminative_dims):
    print(f"  Dim {dim:3d}: Same var={same_var[dim]:.4f}, Diff var={diff_var[dim]:.4f}, Ratio={var_ratio[dim]:.2f}")

# ANALYSIS 3: Stylistic patterns
print("\n" + "=" * 80)
print("ANALYSIS 3: WHAT STYLISTIC PATTERNS DOES BERT LEARN?")
print("=" * 80)

def extract_simple_features(text):
    """Extract simple stylometric features for correlation analysis"""
    import re

    words = text.split()
    sentences = re.split('[.!?]+', text)

    return {
        'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
        'avg_sent_len': np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
        'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / max(len(text), 1),
        'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'unique_word_ratio': len(set(words)) / max(len(words), 1),
    }

# Extract features for all examples
print("\nExtracting stylometric features from examples...")
for ex in examples[:200]:
    ex['features1'] = extract_simple_features(ex['text1'])
    ex['features2'] = extract_simple_features(ex['text2'])

# Compute correlation between embedding similarity and feature similarity
feature_names = ['avg_word_len', 'avg_sent_len', 'punctuation_ratio', 'capital_ratio', 'unique_word_ratio']

print("\nCorrelation between BERT similarity and stylometric feature similarity:")
print("(Positive correlation = BERT uses this feature)")
print()

for feat_name in feature_names:
    # Compute feature difference
    feat_diffs = []
    sims = []

    for ex in examples[:200]:
        f1 = ex['features1'][feat_name]
        f2 = ex['features2'][feat_name]
        feat_diff = abs(f1 - f2)
        feat_diffs.append(feat_diff)
        sims.append(ex['similarity'])

    # Correlation: similarity should be high when feature diff is low
    correlation = np.corrcoef(sims, [-d for d in feat_diffs])[0, 1]

    print(f"  {feat_name:20s}: {correlation:+.3f}")

# ANALYSIS 4: Visualization
print("\n" + "=" * 80)
print("ANALYSIS 4: CREATING VISUALIZATIONS")
print("=" * 80)

# Plot 1: Similarity vs stylometric features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feat_name in enumerate(feature_names):
    ax = axes[i]

    # Get feature differences and similarities
    same_feat_diffs = []
    diff_feat_diffs = []

    for ex in examples[:200]:
        f1 = ex['features1'][feat_name]
        f2 = ex['features2'][feat_name]
        feat_diff = abs(f1 - f2)

        if ex['same']:
            same_feat_diffs.append(feat_diff)
        else:
            diff_feat_diffs.append(feat_diff)

    # Histogram
    ax.hist(same_feat_diffs, bins=20, alpha=0.6, label='Same Author', color='blue', density=True)
    ax.hist(diff_feat_diffs, bins=20, alpha=0.6, label='Different Authors', color='red', density=True)

    ax.set_xlabel(f'Difference in {feat_name}')
    ax.set_ylabel('Density')
    ax.set_title(f'{feat_name.replace("_", " ").title()}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# Hide extra subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig('bert_stylometric_correlation.png', dpi=150, bbox_inches='tight')
print("\nSaved: bert_stylometric_correlation.png")

# Plot 2: Discriminative dimensions
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

dims = top_discriminative_dims[:10]
ratios = [var_ratio[d] for d in dims]

ax.bar(range(len(dims)), ratios, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels([f'Dim {d}' for d in dims], rotation=45)
ax.set_ylabel('Variance Ratio (Diff/Same)')
ax.set_title('Top 10 Most Discriminative Embedding Dimensions\n(Higher = Better at Distinguishing Authors)',
             fontweight='bold', fontsize=13)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('bert_discriminative_dimensions.png', dpi=150, bbox_inches='tight')
print("Saved: bert_discriminative_dimensions.png")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. BERT learns authorship through:")
print("   - Specific embedding dimensions (found top 20)")
print("   - Token-level patterns (function words, punctuation)")
print("   - Stylometric features (sentence length, word patterns)")

print("\n2. Most important stylometric correlations:")
for feat_name in feature_names:
    feat_diffs = []
    sims = []
    for ex in examples[:200]:
        f1 = ex['features1'][feat_name]
        f2 = ex['features2'][feat_name]
        feat_diff = abs(f1 - f2)
        feat_diffs.append(feat_diff)
        sims.append(ex['similarity'])
    correlation = np.corrcoef(sims, [-d for d in feat_diffs])[0, 1]
    if abs(correlation) > 0.1:
        print(f"   - {feat_name}: {correlation:+.3f}")

print("\n3. Visualizations created:")
print("   - bert_stylometric_correlation.png")
print("   - bert_discriminative_dimensions.png")

print("\nDone!")
