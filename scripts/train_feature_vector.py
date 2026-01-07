#!/usr/bin/env python3
"""Train feature vector model for authorship verification."""

import sys
sys.path.insert(0, '/Users/swan/Documents/GitHub/authorship-verification/featurevector')

import pickle
import numpy as np
from tqdm.auto import tqdm
from features import get_transformer, prepare_entry
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from huggingface_hub import hf_hub_download
from datasets import load_dataset

print("=" * 80)
print("TRAINING FEATURE VECTOR MODEL")
print("=" * 80)

# Load datasets
print("\n1. Loading datasets...")
dataset = load_dataset('swan07/authorship-verification')
train_data = dataset['train']
val_data = dataset['val']
print(f"   Train: {len(train_data)} samples")
print(f"   Val: {len(val_data)} samples")

# Fit transformers on subset
print("\n2. Fitting feature transformers (5% sample)...")
docs_1, docs_2 = [], []
for i, ex in enumerate(tqdm(train_data)):
    if np.random.rand() < 0.05:  # 5% sample
        docs_1.append(prepare_entry(ex['text1'], mode='fast', tokenizer='casual'))
        docs_2.append(prepare_entry(ex['text2'], mode='fast', tokenizer='casual'))
    if len(docs_1) >= 5000:  # Max 5000 samples
        break

transformer = get_transformer()
scaler = StandardScaler()
secondary_scaler = StandardScaler()

X = transformer.fit_transform(docs_1 + docs_2).todense()
X = np.asarray(X)
X = scaler.fit_transform(X)
X1, X2 = X[:len(docs_1)], X[len(docs_1):]
secondary_scaler.fit(np.abs(X1 - X2))

print(f"   Features: {X.shape[1]}")

# Train classifier
print("\n3. Training SGD classifier...")
clf = SGDClassifier(loss='log_loss', alpha=0.0002, max_iter=50)

batch_size = 1000
for epoch in range(10):
    print(f"\n   Epoch {epoch+1}/10")
    for i in tqdm(range(0, min(10000, len(train_data)), batch_size)):
        X_batch, y_batch = [], []
        for ex in train_data.select(range(i, min(i+batch_size, len(train_data)))):
            try:
                d1 = prepare_entry(ex['text1'], mode='fast', tokenizer='casual')
                d2 = prepare_entry(ex['text2'], mode='fast', tokenizer='casual')
                if not d1 or not d2: continue

                x1 = transformer.transform([d1]).todense()
                x2 = transformer.transform([d2]).todense()
                x1 = scaler.transform(np.asarray(x1))
                x2 = scaler.transform(np.asarray(x2))
                X_batch.append(secondary_scaler.transform(np.abs(x1 - x2))[0])
                y_batch.append(ex['same'])
            except:
                continue

        if X_batch:
            clf.partial_fit(np.array(X_batch), y_batch, classes=[0, 1])

    # Evaluate on val
    val_probs, val_labels = [], []
    for ex in tqdm(val_data.select(range(min(500, len(val_data))))):
        try:
            d1 = prepare_entry(ex['text1'], mode='fast', tokenizer='casual')
            d2 = prepare_entry(ex['text2'], mode='fast', tokenizer='casual')
            if not d1 or not d2: continue

            x1 = transformer.transform([d1]).todense()
            x2 = transformer.transform([d2]).todense()
            x1 = scaler.transform(np.asarray(x1))
            x2 = scaler.transform(np.asarray(x2))
            X_test = secondary_scaler.transform(np.abs(x1 - x2))
            val_probs.append(clf.predict_proba(X_test)[0, 1])
            val_labels.append(ex['same'])
        except:
            continue

    if val_probs:
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        print(f"   Val AUC: {auc(fpr, tpr):.3f}")

# Save model
print("\n4. Saving model...")
with open('feature_vector_model.pkl', 'wb') as f:
    pickle.dump({
        'clf': clf,
        'transformer': transformer,
        'scaler': scaler,
        'secondary_scaler': secondary_scaler
    }, f)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("Model saved to: feature_vector_model.pkl")
print("=" * 80)
