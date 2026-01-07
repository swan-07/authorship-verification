import os
import sys
sys.path.append('featurevector')

import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from features import prepare_entry, get_transformer

print("=" * 80)
print("FEATURE VECTOR MODEL (Following Paper Methodology)")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use 50K pairs as requested
TRAIN_SIZE = 50000
train_data = dataset['train'].select(range(TRAIN_SIZE))
val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(1000))

print(f"Train: {len(train_data):,} pairs")
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Extract features from training pairs
print("\n2. Preprocessing and extracting features from training...")
print("   (Tokenization, POS tagging, chunking...)")

train_entries = []
for i, ex in enumerate(train_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    train_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 1000 == 0:
        print(f"   {i+1}/{len(train_data)}")

print("   Training pairs preprocessed!")

# Fit transformer on individual texts
print("\n3. Fitting feature transformer...")
all_train_texts = []
for e1, e2, label in train_entries:
    all_train_texts.append(e1)
    all_train_texts.append(e2)

transformer = get_transformer()
transformer.fit(all_train_texts)
print(f"   Transformer fitted!")

# Extract features and compute differences (as per paper)
print("\n4. Extracting feature vectors and computing differences...")

# First, extract all feature vectors
print("   Computing raw feature vectors...")
train_features = []
for i, (e1, e2, label) in enumerate(train_entries):
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]
    train_features.append((f1, f2, label))

    if (i + 1) % 1000 == 0:
        print(f"   {i+1}/{len(train_entries)}")

print(f"   Feature vector size: {len(train_features[0][0])}")

# Standardize features (as per paper)
print("\n5. Standardizing features...")
all_features = []
for f1, f2, label in train_features:
    all_features.append(f1)
    all_features.append(f2)
all_features = np.array(all_features)

feature_scaler = StandardScaler()
feature_scaler.fit(all_features)

# Compute absolute differences and standardize them (as per paper)
print("\n6. Computing absolute differences and standardizing...")
X_train_diff = []
y_train = []

for f1, f2, label in train_features:
    f1_std = feature_scaler.transform([f1])[0]
    f2_std = feature_scaler.transform([f2])[0]
    diff = np.abs(f1_std - f2_std)
    X_train_diff.append(diff)
    y_train.append(label)

X_train_diff = np.array(X_train_diff)
y_train = np.array(y_train)

# Standardize the differences
diff_scaler = StandardScaler()
X_train = diff_scaler.fit_transform(X_train_diff)

print(f"   Final training matrix: {X_train.shape}")

# Train classifier (SGDClassifier with log loss as per paper)
print("\n7. Training SGDClassifier (log loss)...")
classifier = SGDClassifier(
    loss='log_loss',  # Logistic regression with SGD
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
classifier.fit(X_train, y_train)
print("   Classifier trained!")

# Save models
print("\n8. Saving models...")
with open('feature_vector_paper_transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)
with open('feature_vector_paper_feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
with open('feature_vector_paper_diff_scaler.pkl', 'wb') as f:
    pickle.dump(diff_scaler, f)
with open('feature_vector_paper_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print("   Saved all models!")

# Validate
print("\n9. Evaluating on validation set...")
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

    # Standardize features
    f1_std = feature_scaler.transform([f1])[0]
    f2_std = feature_scaler.transform([f2])[0]

    # Compute absolute difference
    diff = np.abs(f1_std - f2_std)

    X_val.append(diff)
    y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Standardize differences
X_val = diff_scaler.transform(X_val)

# Predict
val_probs = classifier.predict_proba(X_val)[:, 1]
val_preds = classifier.predict(X_val)

val_acc = accuracy_score(y_val, val_preds)
val_prec = precision_score(y_val, val_preds)
val_rec = recall_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_probs)

print(f"\n   Validation Results:")
print(f"   Accuracy:  {val_acc*100:.1f}%")
print(f"   Precision: {val_prec*100:.1f}%")
print(f"   Recall:    {val_rec*100:.1f}%")
print(f"   F1:        {val_f1*100:.1f}%")
print(f"   AUC:       {val_auc:.3f}")

# Test
print("\n10. Testing on test set...")
test_entries = []
for i, ex in enumerate(test_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    test_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 200 == 0:
        print(f"   {i+1}/{len(test_data)}")

X_test = []
y_test = []
for e1, e2, label in test_entries:
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]

    f1_std = feature_scaler.transform([f1])[0]
    f2_std = feature_scaler.transform([f2])[0]

    diff = np.abs(f1_std - f2_std)

    X_test.append(diff)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = diff_scaler.transform(X_test)

test_probs = classifier.predict_proba(X_test)[:, 1]
test_preds = classifier.predict(X_test)

accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_probs)

print("\n" + "=" * 80)
print("RESULTS: FEATURE VECTOR MODEL (Paper Method)")
print("=" * 80)
print(f"Training size: {TRAIN_SIZE:,} pairs")
print(f"Test size: {len(test_data):,} pairs")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON TO OTHER MODELS")
print("=" * 80)
print("Paper's Feature Vector (Evaluation): AUC=0.646, F1=0.653")
print(f"Our Feature Vector (Test):           AUC={auc:.3f}, F1={f1:.3f}")
print()
print("Fine-tuned BERT + Calibration:       Acc=70.1%, F1=71.6%, AUC=0.760")
print(f"Feature Vector Model (This):         Acc={accuracy*100:.1f}%, F1={f1*100:.1f}%, AUC={auc:.3f}")

print("\nDone!")
