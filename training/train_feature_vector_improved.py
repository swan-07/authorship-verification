import os
import sys
sys.path.append('featurevector')

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from features import prepare_entry, get_transformer

print("=" * 80)
print("IMPROVED FEATURE VECTOR MODEL")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Use 100K pairs - more than before
TRAIN_SIZE = 100000
train_data = dataset['train'].select(range(TRAIN_SIZE))
val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(1000))

print(f"Train: {len(train_data):,} pairs")
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Extract features from training pairs
print("\n2. Preprocessing and extracting features from training...")
print("   (This will take a while with 100K pairs...)")

train_entries = []
for i, ex in enumerate(train_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    train_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 5000 == 0:
        print(f"   {i+1}/{len(train_data)}")

print("   Training pairs preprocessed!")

# Fit transformer
print("\n3. Fitting feature transformer...")
all_train_texts = []
for e1, e2, label in train_entries:
    all_train_texts.append(e1)
    all_train_texts.append(e2)

transformer = get_transformer()
transformer.fit(all_train_texts)
print(f"   Transformer fitted!")

# Extract features
print("\n4. Extracting feature vectors...")

train_features = []
for i, (e1, e2, label) in enumerate(train_entries):
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]
    train_features.append((f1, f2, label))

    if (i + 1) % 5000 == 0:
        print(f"   {i+1}/{len(train_entries)}")

print(f"   Feature vector size: {len(train_features[0][0])}")

# Standardize features
print("\n5. Standardizing features...")
all_features = []
for f1, f2, label in train_features:
    all_features.append(f1)
    all_features.append(f2)
all_features = np.array(all_features)

feature_scaler = StandardScaler()
feature_scaler.fit(all_features)

# Compute differences with MULTIPLE methods (not just absolute)
print("\n6. Computing feature differences (multiple methods)...")
X_train_list = []
y_train = []

for f1, f2, label in train_features:
    f1_std = feature_scaler.transform([f1])[0]
    f2_std = feature_scaler.transform([f2])[0]

    # Try multiple difference methods as per some papers
    abs_diff = np.abs(f1_std - f2_std)
    product = f1_std * f2_std

    # Combine them (this might capture more information)
    combined = np.concatenate([abs_diff, product])

    X_train_list.append(combined)
    y_train.append(label)

X_train_raw = np.array(X_train_list)
y_train = np.array(y_train)

print(f"   Combined feature size: {X_train_raw.shape[1]}")

# Feature selection - remove low variance and select top K features
print("\n7. Applying feature selection...")

# Remove low-variance features
var_threshold = VarianceThreshold(threshold=0.01)
X_train_var = var_threshold.fit_transform(X_train_raw)
print(f"   After variance threshold: {X_train_var.shape[1]} features")

# Select top K most informative features
k = min(2000, X_train_var.shape[1])  # Top 2000 features or all if less
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train_var, y_train)
print(f"   After selecting top {k}: {X_train_selected.shape[1]} features")

# Final standardization
diff_scaler = StandardScaler()
X_train = diff_scaler.fit_transform(X_train_selected)

print(f"   Final training matrix: {X_train.shape}")

# Train SVM classifier (instead of SGDClassifier)
print("\n8. Training SVM classifier (RBF kernel)...")
print("   (This may take a while...)")
classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,  # Enable probability estimates
    random_state=42,
    max_iter=1000,
    verbose=True
)
classifier.fit(X_train, y_train)
print("   Classifier trained!")

# Save models
print("\n9. Saving models...")
with open('feature_vector_improved_transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)
with open('feature_vector_improved_feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
with open('feature_vector_improved_var_threshold.pkl', 'wb') as f:
    pickle.dump(var_threshold, f)
with open('feature_vector_improved_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
with open('feature_vector_improved_diff_scaler.pkl', 'wb') as f:
    pickle.dump(diff_scaler, f)
with open('feature_vector_improved_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
print("   Saved all models!")

# Validate
print("\n10. Evaluating on validation set...")
val_entries = []
for i, ex in enumerate(val_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    val_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 1000 == 0:
        print(f"   {i+1}/{len(val_data)}")

X_val = []
y_val = []
for e1, e2, label in val_entries:
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]

    f1_std = feature_scaler.transform([f1])[0]
    f2_std = feature_scaler.transform([f2])[0]

    abs_diff = np.abs(f1_std - f2_std)
    product = f1_std * f2_std
    combined = np.concatenate([abs_diff, product])

    X_val.append(combined)
    y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Apply same transformations
X_val = var_threshold.transform(X_val)
X_val = selector.transform(X_val)
X_val = diff_scaler.transform(X_val)

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
print("\n11. Testing on test set...")
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

    abs_diff = np.abs(f1_std - f2_std)
    product = f1_std * f2_std
    combined = np.concatenate([abs_diff, product])

    X_test.append(combined)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = var_threshold.transform(X_test)
X_test = selector.transform(X_test)
X_test = diff_scaler.transform(X_test)

test_probs = classifier.predict_proba(X_test)[:, 1]
test_preds = classifier.predict(X_test)

accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_probs)

print("\n" + "=" * 80)
print("RESULTS: IMPROVED FEATURE VECTOR MODEL")
print("=" * 80)
print(f"Training size: {TRAIN_SIZE:,} pairs")
print(f"Features used: {X_train.shape[1]:,}")
print(f"Test size: {len(test_data):,} pairs")
print(f"Accuracy:  {accuracy*100:.1f}%")
print(f"Precision: {precision*100:.1f}%")
print(f"Recall:    {recall*100:.1f}%")
print(f"F1 Score:  {f1*100:.1f}%")
print(f"AUC:       {auc:.3f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("Original Feature Vector (50K):   Acc=58.6%, F1=57.9%, AUC=0.619")
print(f"Improved Feature Vector (100K):  Acc={accuracy*100:.1f}%, F1={f1*100:.1f}%, AUC={auc:.3f}")
print()
print("Fine-tuned BERT:                 Acc=70.1%, F1=71.6%, AUC=0.760")

print("\nDone!")
