import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

print("=" * 80)
print("DETAILED ENSEMBLE ANALYSIS")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use larger sets
val_data = dataset['validation'].select(range(10000))  # 10K for training combiner
test_data = dataset['test'].select(range(10000))  # 10K for testing

print(f"Validation: {len(val_data):,} pairs (for training ensemble)")
print(f"Test: {len(test_data):,} pairs (for evaluation)")

# Load models
print("\n2. Loading models...")
bert_model = SentenceTransformer('bert-finetuned-authorship')
with open('stylometric_pan_model.pkl', 'rb') as f:
    stylo_data = pickle.load(f)

char_vectorizers = stylo_data['char_vectorizers']
word_vectorizer = stylo_data['word_vectorizer']
punct_vectorizer = stylo_data['punct_vectorizer']
stylo_classifier = stylo_data['classifier']
print("   ✓ Models loaded")

def extract_punctuation(text):
    return re.sub(r'[^\s!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)

def get_stylometric_features(text1, text2):
    """Get all 8 stylometric features"""
    features = []
    feature_names = []
    
    # Char n-gram similarities
    for n in [3, 4, 5]:
        v1 = char_vectorizers[n].transform([text1])
        v2 = char_vectorizers[n].transform([text2])
        sim = cosine_similarity(v1, v2)[0, 0]
        features.append(sim)
        feature_names.append(f'Char {n}-gram')
    
    # Word n-gram similarity
    v1 = word_vectorizer.transform([text1])
    v2 = word_vectorizer.transform([text2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)
    feature_names.append('Word n-gram')
    
    # Punctuation n-gram similarity
    p1 = extract_punctuation(text1)
    p2 = extract_punctuation(text2)
    v1 = punct_vectorizer.transform([p1])
    v2 = punct_vectorizer.transform([p2])
    sim = cosine_similarity(v1, v2)[0, 0]
    features.append(sim)
    feature_names.append('Punctuation')
    
    # Length ratio
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    features.append(len_ratio)
    feature_names.append('Length ratio')
    
    # Sentence ratio
    sent1 = len(text1.split('.'))
    sent2 = len(text2.split('.'))
    sent_ratio = min(sent1, sent2) / max(sent1, sent2) if max(sent1, sent2) > 0 else 1.0
    features.append(sent_ratio)
    feature_names.append('Sentence ratio')
    
    # Average word length
    words1 = text1.split()
    words2 = text2.split()
    avg_len1 = np.mean([len(w) for w in words1]) if words1 else 0
    avg_len2 = np.mean([len(w) for w in words2]) if words2 else 0
    len_sim = 1 - abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2) if max(avg_len1, avg_len2) > 0 else 1.0
    features.append(len_sim)
    feature_names.append('Avg word length')
    
    return features, feature_names

def get_all_scores(text1, text2):
    """Get both BERT and stylometric scores with details"""
    # BERT
    emb1 = bert_model.encode(text1, convert_to_numpy=True)
    emb2 = bert_model.encode(text2, convert_to_numpy=True)
    bert_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    # Stylometric
    stylo_features, feature_names = get_stylometric_features(text1, text2)
    stylo_score = stylo_classifier.predict_proba([stylo_features])[0, 1]
    
    return bert_score, stylo_score, stylo_features, feature_names

# Train ensemble on validation set
print("\n3. Training ensemble on 10K validation pairs...")
val_bert_scores = []
val_stylo_scores = []
val_labels = []

for i, ex in enumerate(val_data):
    bert_score, stylo_score, _, _ = get_all_scores(ex['text1'], ex['text2'])
    val_bert_scores.append(bert_score)
    val_stylo_scores.append(stylo_score)
    val_labels.append(ex['same'])
    
    if (i + 1) % 2000 == 0:
        print(f"   {i+1:,}/{len(val_data):,}")

X_val = np.column_stack([np.array(val_bert_scores), np.array(val_stylo_scores)])
ensemble_combiner = LogisticRegression(random_state=42)
ensemble_combiner.fit(X_val, val_labels)

bert_weight = ensemble_combiner.coef_[0][0]
stylo_weight = ensemble_combiner.coef_[0][1]
bias = ensemble_combiner.intercept_[0]

print(f"   ✓ Trained")
print(f"   Ensemble formula: P = σ({bert_weight:.3f}×BERT + {stylo_weight:.3f}×Stylo + {bias:.3f})")

# Show detailed example
print("\n" + "=" * 80)
print("DETAILED EXAMPLE")
print("=" * 80)

example = test_data[0]
print(f"\nTrue label: {'SAME AUTHOR' if example['same'] else 'DIFFERENT AUTHORS'}")
print(f"\nText 1 (first 200 chars):\n{example['text1'][:200]}...")
print(f"\nText 2 (first 200 chars):\n{example['text2'][:200]}...")

bert_score, stylo_score, stylo_features, feature_names = get_all_scores(example['text1'], example['text2'])

print(f"\n--- BERT Analysis ---")
print(f"Cosine similarity: {bert_score:.4f}")
print(f"Interpretation: {'High similarity' if bert_score > 0.5 else 'Low similarity'}")

print(f"\n--- Stylometric Analysis ---")
for fname, fval in zip(feature_names, stylo_features):
    print(f"  {fname:20s}: {fval:.4f}")
print(f"Stylometric score: {stylo_score:.4f}")

print(f"\n--- Ensemble Combination ---")
print(f"BERT score:        {bert_score:.4f} × {bert_weight:.3f} = {bert_score * bert_weight:.4f}")
print(f"Stylo score:       {stylo_score:.4f} × {stylo_weight:.3f} = {stylo_score * stylo_weight:.4f}")
print(f"Bias:              {bias:.4f}")
print(f"Sum:               {bert_score * bert_weight + stylo_score * stylo_weight + bias:.4f}")

ensemble_prob = ensemble_combiner.predict_proba([[bert_score, stylo_score]])[0, 1]
print(f"After sigmoid:     {ensemble_prob:.4f}")
print(f"\nFinal prediction: {'SAME AUTHOR' if ensemble_prob >= 0.5 else 'DIFFERENT AUTHORS'}")
print(f"Correct: {'✓' if (ensemble_prob >= 0.5) == example['same'] else '✗'}")

# Evaluate on larger test set
print("\n" + "=" * 80)
print("EVALUATION ON 10K TEST SET")
print("=" * 80)

test_bert_scores = []
test_stylo_scores = []
test_labels = []

print("\nComputing predictions...")
for i, ex in enumerate(test_data):
    bert_score, stylo_score, _, _ = get_all_scores(ex['text1'], ex['text2'])
    test_bert_scores.append(bert_score)
    test_stylo_scores.append(stylo_score)
    test_labels.append(ex['same'])
    
    if (i + 1) % 2000 == 0:
        print(f"   {i+1:,}/{len(test_data):,}")

test_bert_scores = np.array(test_bert_scores)
test_stylo_scores = np.array(test_stylo_scores)
test_labels = np.array(test_labels)

# Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# BERT
bert_preds = (test_bert_scores >= 0.5).astype(int)
bert_acc = accuracy_score(test_labels, bert_preds)
bert_f1 = f1_score(test_labels, bert_preds)
bert_auc = roc_auc_score(test_labels, test_bert_scores)
print(f"\nBERT:          Acc={bert_acc*100:.1f}%, F1={bert_f1*100:.1f}%, AUC={bert_auc:.3f}")

# Stylometric
stylo_preds = (test_stylo_scores >= 0.5).astype(int)
stylo_acc = accuracy_score(test_labels, stylo_preds)
stylo_f1 = f1_score(test_labels, stylo_preds)
stylo_auc = roc_auc_score(test_labels, test_stylo_scores)
print(f"Stylometric:   Acc={stylo_acc*100:.1f}%, F1={stylo_f1*100:.1f}%, AUC={stylo_auc:.3f}")

# Ensemble
X_test = np.column_stack([test_bert_scores, test_stylo_scores])
ensemble_probs = ensemble_combiner.predict_proba(X_test)[:, 1]
ensemble_preds = (ensemble_probs >= 0.5).astype(int)
ensemble_acc = accuracy_score(test_labels, ensemble_preds)
ensemble_f1 = f1_score(test_labels, ensemble_preds)
ensemble_auc = roc_auc_score(test_labels, ensemble_probs)
print(f"ENSEMBLE:      Acc={ensemble_acc*100:.1f}%, F1={ensemble_f1*100:.1f}%, AUC={ensemble_auc:.3f}")

print(f"\n✓ Improvement over BERT: +{(ensemble_acc - bert_acc)*100:.1f}% accuracy")
print(f"✓ Improvement over best single model: +{(ensemble_acc - max(bert_acc, stylo_acc))*100:.1f}%")
