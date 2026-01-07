import os
import sys
sys.path.append('featurevector')

import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from features import prepare_entry
import time
import gc

print("=" * 80)
print("MEMORY-EFFICIENT FEATURE VECTOR MODEL")
print("=" * 80)

# Configuration
CHECKPOINT_DIR = 'feature_vector_checkpoints'

# Load dataset metadata only
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(1000))

print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Load transformer
print("\n2. Loading transformer...")
transformer_file = os.path.join(CHECKPOINT_DIR, 'transformer.pkl')
with open(transformer_file, 'rb') as f:
    transformer = pickle.load(f)
print("   ✓ Transformer loaded")

# Check if scalers already computed
scaler_file = os.path.join(CHECKPOINT_DIR, 'scalers.pkl')

if os.path.exists(scaler_file):
    print("\n3. Loading saved scalers and training data...")
    with open(scaler_file, 'rb') as f:
        scaler_data = pickle.load(f)

    feature_scaler = scaler_data['feature_scaler']
    var_threshold = scaler_data['var_threshold']
    selector = scaler_data['selector']
    diff_scaler = scaler_data['diff_scaler']
    X_train = scaler_data['X_train']
    y_train = scaler_data['y_train']

    print(f"   ✓ Loaded training matrix: {X_train.shape}")

else:
    # Load features in STREAMING mode to avoid memory explosion
    print("\n3. Loading features file (16GB - this will take a while)...")
    features_file = os.path.join(CHECKPOINT_DIR, 'features.pkl')

    with open(features_file, 'rb') as f:
        feature_data = pickle.load(f)

    train_features = feature_data['train_features']
    n_samples = len(train_features)
    print(f"   ✓ Loaded {n_samples:,} feature pairs")
    print(f"   Feature vector size: {len(train_features[0][0])}")

    # Step 1: Fit feature scaler (process in batches to avoid memory)
    print("\n4. Fitting feature scaler on all individual texts...")
    print("   Processing in batches to avoid memory issues...")

    # We need to compute statistics incrementally
    # For StandardScaler, we need mean and variance
    # We'll use Welford's online algorithm

    n_features = len(train_features[0][0])
    n_texts = n_samples * 2

    # First pass: compute mean
    sum_features = np.zeros(n_features, dtype=np.float64)
    count = 0

    batch_size = 5000
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        for f1, f2, _ in train_features[i:batch_end]:
            sum_features += f1
            sum_features += f2
            count += 2

        if (i + batch_size) % 10000 == 0:
            print(f"   Mean pass: {i + batch_size}/{n_samples}", end='\r')
            sys.stdout.flush()

    mean_features = sum_features / count
    print(f"\n   ✓ Computed mean")

    # Second pass: compute variance
    sum_sq_diff = np.zeros(n_features, dtype=np.float64)

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        for f1, f2, _ in train_features[i:batch_end]:
            sum_sq_diff += (f1 - mean_features) ** 2
            sum_sq_diff += (f2 - mean_features) ** 2

        if (i + batch_size) % 10000 == 0:
            print(f"   Variance pass: {i + batch_size}/{n_samples}", end='\r')
            sys.stdout.flush()

    var_features = sum_sq_diff / count
    std_features = np.sqrt(var_features)

    # Create StandardScaler and set parameters manually
    feature_scaler = StandardScaler()
    feature_scaler.mean_ = mean_features
    feature_scaler.scale_ = std_features
    feature_scaler.var_ = var_features
    feature_scaler.n_features_in_ = n_features
    feature_scaler.n_samples_seen_ = count

    print(f"\n   ✓ Feature scaler fitted")

    # Step 2: Compute difference vectors in batches
    print("\n5. Computing difference vectors in batches...")

    X_train_list = []
    y_train = []

    for i, (f1, f2, label) in enumerate(train_features):
        # Standardize
        f1_std = (f1 - mean_features) / std_features
        f2_std = (f2 - mean_features) / std_features

        # Compute differences
        abs_diff = np.abs(f1_std - f2_std)
        product = f1_std * f2_std
        combined = np.concatenate([abs_diff, product])

        X_train_list.append(combined)
        y_train.append(label)

        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{n_samples}", end='\r')
            sys.stdout.flush()

        # Periodically force garbage collection
        if (i + 1) % 10000 == 0:
            gc.collect()

    X_train_raw = np.array(X_train_list)
    y_train = np.array(y_train)

    # Can delete train_features now
    del train_features
    del X_train_list
    gc.collect()

    print(f"\n   ✓ Difference vectors computed: {X_train_raw.shape}")

    # Step 3: Feature selection
    print("\n6. Feature selection...")

    var_threshold = VarianceThreshold(threshold=0.01)
    X_train_var = var_threshold.fit_transform(X_train_raw)
    print(f"   After variance threshold: {X_train_var.shape[1]} features")

    del X_train_raw
    gc.collect()

    k = min(2000, X_train_var.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_var, y_train)
    print(f"   After selecting top {k}: {X_train_selected.shape[1]} features")

    del X_train_var
    gc.collect()

    # Step 4: Final standardization
    diff_scaler = StandardScaler()
    X_train = diff_scaler.fit_transform(X_train_selected)

    del X_train_selected
    gc.collect()

    print(f"   ✓ Final training matrix: {X_train.shape}")

    # Save everything
    print("\n7. Saving scalers and training matrix...")
    with open(scaler_file, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'var_threshold': var_threshold,
            'selector': selector,
            'diff_scaler': diff_scaler,
            'X_train': X_train,
            'y_train': y_train
        }, f)

    print(f"   ✓ Saved to {scaler_file}")

# Train classifier
classifier_file = os.path.join(CHECKPOINT_DIR, 'classifier.pkl')

if os.path.exists(classifier_file):
    print("\n8. Loading trained classifier...")
    with open(classifier_file, 'rb') as f:
        classifier = pickle.load(f)
    print("   ✓ Classifier loaded")
else:
    print("\n8. Training LinearSVC classifier...")
    print("   (LinearSVC is much faster than RBF SVM)")

    start_time = time.time()

    classifier = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=42,
        verbose=1
    )

    classifier.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"\n   ✓ Training complete in {elapsed/60:.1f} minutes")

    # Save classifier
    with open(classifier_file, 'wb') as f:
        pickle.dump(classifier, f)

    print(f"   ✓ Saved to {classifier_file}")

# Helper function to process pairs
def process_pair(e1, e2):
    f1 = transformer.transform([e1]).toarray()[0]
    f2 = transformer.transform([e2]).toarray()[0]

    f1_std = feature_scaler.transform([f1])[0]
    f2_std = feature_scaler.transform([f2])[0]

    abs_diff = np.abs(f1_std - f2_std)
    product = f1_std * f2_std
    combined = np.concatenate([abs_diff, product])

    # Apply transformations
    combined = var_threshold.transform([combined])
    combined = selector.transform(combined)
    combined = diff_scaler.transform(combined)

    return combined[0]

# Validate
print("\n9. Evaluating on validation set...")
val_entries = []
for i, ex in enumerate(val_data):
    entry1 = prepare_entry(ex['text1'], mode='fast')
    entry2 = prepare_entry(ex['text2'], mode='fast')
    val_entries.append((entry1, entry2, ex['same']))

    if (i + 1) % 500 == 0:
        print(f"   {i+1}/{len(val_data)}", end='\r')
        sys.stdout.flush()

X_val = np.array([process_pair(e1, e2) for e1, e2, _ in val_entries])
y_val = np.array([label for _, _, label in val_entries])

val_preds = classifier.predict(X_val)
val_scores = classifier.decision_function(X_val)

val_acc = accuracy_score(y_val, val_preds)
val_prec = precision_score(y_val, val_preds)
val_rec = recall_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_scores)

print(f"\n\n   Validation Results:")
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
        print(f"   {i+1}/{len(test_data)}", end='\r')
        sys.stdout.flush()

X_test = np.array([process_pair(e1, e2) for e1, e2, _ in test_entries])
y_test = np.array([label for _, _, label in test_entries])

test_preds = classifier.predict(X_test)
test_scores = classifier.decision_function(X_test)

accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_scores)

print("\n" + "=" * 80)
print("FINAL RESULTS: MEMORY-EFFICIENT FEATURE VECTOR MODEL")
print("=" * 80)
print(f"Training size: 100,000 pairs")
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

# Save final model for ensemble
print("\n11. Saving final model for ensemble...")
final_model = {
    'transformer': transformer,
    'feature_scaler': feature_scaler,
    'var_threshold': var_threshold,
    'selector': selector,
    'diff_scaler': diff_scaler,
    'classifier': classifier
}

with open('feature_vector_final.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("   ✓ Saved to feature_vector_final.pkl (ready for ensemble)")

print("\n✓ Done! You can now use this model in an ensemble with BERT.")
