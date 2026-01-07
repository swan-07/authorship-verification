import os
import sys
sys.path.append('featurevector')

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from features import prepare_entry, get_transformer

print("=" * 80)
print("TRAINING FEATURE VECTOR MODEL")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use subset for faster training
TRAIN_SIZE = 10000
train_data = dataset['train'].select(range(TRAIN_SIZE))
val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(500))  # Start with small test

print(f"Train: {len(train_data):,} pairs")
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Extract features from text pairs
print("\n2. Extracting features from training set...")
print("   This will take a while (tokenizing, POS tagging, etc.)...")

train_entries = []
for i, ex in enumerate(train_data):
    # Prepare both texts
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    train_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 500 == 0:
        print(f"   {i+1}/{len(train_data)}")

print("   Training features extracted!")

# Fit transformer on individual texts
print("\n3. Fitting feature transformer...")
all_train_texts = []
for e1, e2, label in train_entries:
    all_train_texts.append(e1)
    all_train_texts.append(e2)

transformer = get_transformer()
transformer.fit(all_train_texts)
print(f"   Transformer fitted!")

# Save transformer
with open('feature_transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)
print("   Saved transformer to feature_transformer.pkl")

# Transform pairs to feature vectors
print("\n4. Computing similarity features for training...")
X_train = []
y_train = []

for i, (e1, e2, label) in enumerate(train_entries):
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]

    # Compute similarity features
    diff = np.abs(f1 - f2)
    prod = f1 * f2
    combined = np.concatenate([diff, prod])

    X_train.append(combined)
    y_train.append(label)

    if (i + 1) % 500 == 0:
        print(f"   {i+1}/{len(train_entries)}")

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"   Training matrix: {X_train.shape}")

# Train classifier
print("\n5. Training logistic regression classifier...")
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)
print("   Classifier trained!")

# Save classifier
with open('feature_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print("   Saved to feature_classifier.pkl")

# Validate
print("\n6. Validating on validation set...")
val_entries = []
for i, ex in enumerate(val_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    val_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 500 == 0:
        print(f"   {i+1}/{len(val_data)}")

X_val = []
y_val = []
for e1, e2, label in val_entries:
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]
    diff = np.abs(f1 - f2)
    prod = f1 * f2
    combined = np.concatenate([diff, prod])
    X_val.append(combined)
    y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

val_probs = classifier.predict_proba(X_val)[:, 1]
val_preds = classifier.predict(X_val)

val_acc = accuracy_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_probs)

print(f"\n   Validation Accuracy: {val_acc*100:.1f}%")
print(f"   Validation AUC: {val_auc:.3f}")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_val, val_probs)
best_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[best_idx]
print(f"   Optimal threshold: {optimal_threshold:.4f}")

# Test
print("\n7. Testing on test set...")
test_entries = []
for i, ex in enumerate(test_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    test_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 100 == 0:
        print(f"   {i+1}/{len(test_data)}")

X_test = []
y_test = []
for e1, e2, label in test_entries:
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]
    diff = np.abs(f1 - f2)
    prod = f1 * f2
    combined = np.concatenate([diff, prod])
    X_test.append(combined)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

test_probs = classifier.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= optimal_threshold).astype(int)

accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_probs)

print("\n" + "=" * 80)
print("RESULTS: FEATURE VECTOR MODEL")
print("=" * 80)
print(f"Training size: {TRAIN_SIZE:,} pairs")
print(f"Features: {X_train.shape[1]:,}")
print(f"Test size: {len(test_data):,} pairs")
print(f"Threshold: {optimal_threshold:.4f}")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON TO OTHER MODELS")
print("=" * 80)
print("Base BERT + Calibration:       Acc=63.7%, F1=58.2%, AUC=0.676")
print("MPNet + Calibration:           Acc=56.0%, F1=44.7%, AUC=0.588")
print("Fine-tuned BERT + Calibration: Acc=70.1%, F1=71.6%, AUC=0.760")
print(f"Feature Vector Model:          Acc={accuracy*100:.1f}%, F1={f1*100:.1f}%, AUC={auc:.3f}")

print("\nDone!")
