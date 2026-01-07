import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import sys

print("=" * 80)
print("STYLOMETRIC MODEL - PAN WINNING STRATEGIES")
print("Character N-grams + TF-IDF + Cosine Similarity")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use 50K pairs for training (balance between performance and memory)
TRAIN_SIZE = 50000
train_data = dataset['train'].select(range(TRAIN_SIZE))
val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(1000))

print(f"Training: {len(train_data):,} pairs")
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Strategy 1: Character n-grams (3-5) with TF-IDF
print("\n2. Creating character n-gram models...")

# Multiple n-gram ranges (PAN baselines use char 3-grams, we'll try 3-5)
vectorizers = {}
for n in [3, 4, 5]:
    print(f"   Training char {n}-gram vectorizer...")
    vectorizers[n] = TfidfVectorizer(
        analyzer='char',
        ngram_range=(n, n),
        max_features=5000,  # Balanced features
        lowercase=False,    # Keep case for stylometry
        min_df=2
    )

    # Fit on all training texts
    all_texts = []
    for ex in train_data:
        all_texts.append(ex['text1'])
        all_texts.append(ex['text2'])

    vectorizers[n].fit(all_texts)
    print(f"      ✓ Char {n}-gram: {len(vectorizers[n].get_feature_names_out())} features")

# Strategy 2: Word n-grams (1-2) with TF-IDF for lexical features
print("\n3. Creating word n-gram model...")
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=3000,  # Balanced features
    lowercase=False,
    min_df=2
)

all_texts = []
for ex in train_data:
    all_texts.append(ex['text1'])
    all_texts.append(ex['text2'])

word_vectorizer.fit(all_texts)
print(f"   ✓ Word n-grams: {len(word_vectorizer.get_feature_names_out())} features")

# Strategy 3: Punctuation n-grams (PAN 2022 winner used these)
print("\n4. Creating punctuation n-gram model...")

import re

def extract_punctuation(text):
    """Extract only punctuation and spaces"""
    return re.sub(r'[^\s!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)

punct_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 3),
    max_features=500,
    lowercase=False,
    min_df=2
)

all_punct = []
for ex in train_data:
    all_punct.append(extract_punctuation(ex['text1']))
    all_punct.append(extract_punctuation(ex['text2']))

punct_vectorizer.fit(all_punct)
print(f"   ✓ Punctuation n-grams: {len(punct_vectorizer.get_feature_names_out())} features")

print("\n5. Computing similarity features for training pairs...")

# For each pair, compute multiple similarity scores
X_train_features = []
y_train = []

for i, ex in enumerate(train_data):
    features = []

    # Char n-gram similarities
    for n in [3, 4, 5]:
        v1 = vectorizers[n].transform([ex['text1']])
        v2 = vectorizers[n].transform([ex['text2']])
        sim = cosine_similarity(v1, v2)[0, 0]
        features.append(sim)

    # Word n-gram similarity
    v1 = word_vectorizer.transform([ex['text1']])
    v2 = word_vectorizer.transform([ex['text2']])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)

    # Punctuation n-gram similarity
    p1 = extract_punctuation(ex['text1'])
    p2 = extract_punctuation(ex['text2'])
    v1 = punct_vectorizer.transform([p1])
    v2 = punct_vectorizer.transform([p2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)

    # Length ratio (simple but effective)
    len_ratio = min(len(ex['text1']), len(ex['text2'])) / max(len(ex['text1']), len(ex['text2']))
    features.append(len_ratio)

    # Sentence length similarity
    sent1 = len(ex['text1'].split('.'))
    sent2 = len(ex['text2'].split('.'))
    sent_ratio = min(sent1, sent2) / max(sent1, sent2) if max(sent1, sent2) > 0 else 1.0
    features.append(sent_ratio)

    # Average word length similarity
    words1 = ex['text1'].split()
    words2 = ex['text2'].split()
    avg_len1 = np.mean([len(w) for w in words1]) if words1 else 0
    avg_len2 = np.mean([len(w) for w in words2]) if words2 else 0
    len_sim = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2) if max(avg_len1, avg_len2) > 0 else 1.0
    features.append(len_sim)

    X_train_features.append(features)
    y_train.append(ex['same'])

    if (i + 1) % 10000 == 0:
        print(f"   {i+1:,}/{len(train_data):,}", end='\r')
        sys.stdout.flush()

X_train = np.array(X_train_features)
y_train = np.array(y_train)

print(f"\n   ✓ Training features: {X_train.shape}")

# Train meta-classifier on similarity features
print("\n6. Training meta-classifier on similarity features...")
# Use class_weight='balanced' to handle any class imbalance
classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
classifier.fit(X_train, y_train)

print("   ✓ Classifier trained")

# Feature importance
print("\n   Feature importance:")
feature_names = [
    'Char 3-gram sim',
    'Char 4-gram sim',
    'Char 5-gram sim',
    'Word n-gram sim',
    'Punctuation sim',
    'Length ratio',
    'Sentence ratio',
    'Avg word len sim'
]

for name, coef in zip(feature_names, classifier.coef_[0]):
    print(f"      {name:20s}: {coef:+.3f}")

# Save model
print("\n7. Saving model...")
model_data = {
    'char_vectorizers': vectorizers,
    'word_vectorizer': word_vectorizer,
    'punct_vectorizer': punct_vectorizer,
    'classifier': classifier
}

with open('stylometric_pan_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   ✓ Model saved to stylometric_pan_model.pkl")

# Helper function
def get_stylometric_score(text1, text2):
    """Compute stylometric similarity score"""
    features = []

    # Char n-gram similarities
    for n in [3, 4, 5]:
        v1 = vectorizers[n].transform([text1])
        v2 = vectorizers[n].transform([text2])
        sim = cosine_similarity(v1, v2)[0, 0]
        features.append(sim)

    # Word n-gram similarity
    v1 = word_vectorizer.transform([text1])
    v2 = word_vectorizer.transform([text2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)

    # Punctuation n-gram similarity
    p1 = extract_punctuation(text1)
    p2 = extract_punctuation(text2)
    v1 = punct_vectorizer.transform([p1])
    v2 = punct_vectorizer.transform([p2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)

    # Length ratio
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    features.append(len_ratio)

    # Sentence ratio
    sent1 = len(text1.split('.'))
    sent2 = len(text2.split('.'))
    sent_ratio = min(sent1, sent2) / max(sent1, sent2) if max(sent1, sent2) > 0 else 1.0
    features.append(sent_ratio)

    # Average word length
    words1 = text1.split()
    words2 = text2.split()
    avg_len1 = np.mean([len(w) for w in words1]) if words1 else 0
    avg_len2 = np.mean([len(w) for w in words2]) if words2 else 0
    len_sim = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2) if max(avg_len1, avg_len2) > 0 else 1.0
    features.append(len_sim)

    prob = classifier.predict_proba([features])[0, 1]
    return prob

# Validate
print("\n8. Evaluating on validation set...")
val_probs = []
val_labels = []

for i, ex in enumerate(val_data):
    prob = get_stylometric_score(ex['text1'], ex['text2'])
    val_probs.append(prob)
    val_labels.append(ex['same'])

    if (i + 1) % 1000 == 0:
        print(f"   {i+1:,}/{len(val_data):,}", end='\r')
        sys.stdout.flush()

val_probs = np.array(val_probs)
val_labels = np.array(val_labels)

# Evaluate with default 0.5 threshold first
default_threshold = 0.5
val_preds_default = (val_probs >= default_threshold).astype(int)

val_acc_default = accuracy_score(val_labels, val_preds_default)
val_prec_default = precision_score(val_labels, val_preds_default)
val_rec_default = recall_score(val_labels, val_preds_default)
val_f1_default = f1_score(val_labels, val_preds_default)
val_auc = roc_auc_score(val_labels, val_probs)

print(f"\n\n   Validation Results (threshold=0.5):")
print(f"   Accuracy:  {val_acc_default*100:.1f}%")
print(f"   Precision: {val_prec_default*100:.1f}%")
print(f"   Recall:    {val_rec_default*100:.1f}%")
print(f"   F1:        {val_f1_default*100:.1f}%")
print(f"   AUC:       {val_auc:.3f}")

# Find optimal threshold for reference (but use 0.5 for test)
thresholds = np.arange(0, 1, 0.01)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Use default 0.5 threshold for final evaluation to avoid overfitting
val_preds = val_preds_default
best_threshold = default_threshold

print(f"\n   Note: Using threshold={best_threshold:.1f} for test set (standard threshold)")

# Test
print("\n9. Testing on test set...")
test_probs = []
test_labels = []

for i, ex in enumerate(test_data):
    prob = get_stylometric_score(ex['text1'], ex['text2'])
    test_probs.append(prob)
    test_labels.append(ex['same'])

    if (i + 1) % 500 == 0:
        print(f"   {i+1:,}/{len(test_data):,}", end='\r')
        sys.stdout.flush()

test_probs = np.array(test_probs)
test_labels = np.array(test_labels)
test_preds = (test_probs >= best_threshold).astype(int)

accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
auc = roc_auc_score(test_labels, test_probs)

print("\n" + "=" * 80)
print("FINAL RESULTS: STYLOMETRIC MODEL")
print("=" * 80)
print(f"Training size: {TRAIN_SIZE:,} pairs")
print(f"Test size: {len(test_data):,} pairs")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Paper Feature Vector (50K):  Acc=58.6%, F1=57.9%, AUC=0.619")
print(f"Stylometric PAN (100K):      Acc={accuracy*100:.1f}%, F1={f1*100:.1f}%, AUC={auc:.3f}")
print()
print("Fine-tuned BERT (50K):       Acc=70.1%, F1=71.6%, AUC=0.760")

print("\n✓ Done! Model saved and ready for ensemble.")
