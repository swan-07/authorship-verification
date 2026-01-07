import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

print("=" * 80)
print("ENSEMBLE MODEL: BERT + STYLOMETRIC")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

test_data = dataset['test'].select(range(1000))
val_data = dataset['validation'].select(range(5000))
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Load BERT model
print("\n2. Loading BERT model...")
bert_model = SentenceTransformer('bert-finetuned-authorship')
print("   ✓ BERT loaded")

# Load stylometric model
print("\n3. Loading stylometric model...")
with open('stylometric_pan_model.pkl', 'rb') as f:
    stylo_data = pickle.load(f)

char_vectorizers = stylo_data['char_vectorizers']
word_vectorizer = stylo_data['word_vectorizer']
punct_vectorizer = stylo_data['punct_vectorizer']
stylo_classifier = stylo_data['classifier']
print("   ✓ Stylometric model loaded")

def extract_punctuation(text):
    """Extract only punctuation and spaces"""
    return re.sub(r'[^\s!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', '', text)

def get_stylometric_score(text1, text2):
    """Compute stylometric similarity score"""
    features = []

    # Char n-gram similarities
    for n in [3, 4, 5]:
        v1 = char_vectorizers[n].transform([text1])
        v2 = char_vectorizers[n].transform([text2])
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

    prob = stylo_classifier.predict_proba([features])[0, 1]
    return prob

# Get predictions on validation set
print("\n4. Getting predictions on validation set...")
val_bert_scores = []
val_stylo_scores = []
val_labels = []

for i, ex in enumerate(val_data):
    # BERT score
    emb1 = bert_model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = bert_model.encode(ex['text2'], convert_to_numpy=True)
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    val_bert_scores.append(cos_sim)

    # Stylometric score
    stylo_score = get_stylometric_score(ex['text1'], ex['text2'])
    val_stylo_scores.append(stylo_score)
    val_labels.append(ex['same'])

    if (i + 1) % 1000 == 0:
        print(f"   {i+1:,}/{len(val_data):,}")

val_bert_scores = np.array(val_bert_scores)
val_stylo_scores = np.array(val_stylo_scores)
val_labels = np.array(val_labels)
print(f"   ✓ Complete")

# Train ensemble
print("\n5. Training ensemble combiner...")
X_val = np.column_stack([val_bert_scores, val_stylo_scores])
ensemble_combiner = LogisticRegression(random_state=42)
ensemble_combiner.fit(X_val, val_labels)
print(f"   ✓ BERT weight={ensemble_combiner.coef_[0][0]:.3f}, Stylo weight={ensemble_combiner.coef_[0][1]:.3f}")

# Evaluate on test
print("\n6. Evaluating on test set...")
test_bert_scores = []
test_stylo_scores = []
test_labels = []

for i, ex in enumerate(test_data):
    emb1 = bert_model.encode(ex['text1'], convert_to_numpy=True)
    emb2 = bert_model.encode(ex['text2'], convert_to_numpy=True)
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    test_bert_scores.append(cos_sim)

    stylo_score = get_stylometric_score(ex['text1'], ex['text2'])
    test_stylo_scores.append(stylo_score)
    test_labels.append(ex['same'])

    if (i + 1) % 500 == 0:
        print(f"   {i+1:,}/{len(test_data):,}")

test_bert_scores = np.array(test_bert_scores)
test_stylo_scores = np.array(test_stylo_scores)
test_labels = np.array(test_labels)
print(f"   ✓ Complete")

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

print(f"\nImprovement: +{(ensemble_acc - bert_acc)*100:.1f}% over BERT")

# Save
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump({'bert_path': 'bert-finetuned-authorship', 'stylo': stylo_data, 'combiner': ensemble_combiner}, f)
print("\n✓ Saved to ensemble_model.pkl")
