import os
import sys
sys.path.append('featurevector')

import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from features import prepare_entry, get_transformer
import time

print("=" * 80)
print("CHECKPOINTED FEATURE VECTOR MODEL")
print("=" * 80)

# Configuration
TRAIN_SIZE = 100000
CHECKPOINT_DIR = 'feature_vector_checkpoints'
BATCH_SIZE = 5000  # Process in batches

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load dataset
print("\n1. Loading dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['train'].select(range(TRAIN_SIZE))
val_data = dataset['validation'].select(range(5000))
test_data = dataset['test'].select(range(1000))

print(f"Train: {len(train_data):,} pairs")
print(f"Validation: {len(val_data):,} pairs")
print(f"Test: {len(test_data):,} pairs")

# Check for existing checkpoints
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'preprocessing_checkpoint.pkl')

if os.path.exists(checkpoint_file):
    print("\n2. RESUMING from checkpoint...")
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)

    train_entries = checkpoint['train_entries']
    last_batch = checkpoint['last_batch']
    transformer = checkpoint.get('transformer', None)

    print(f"   Loaded {len(train_entries)} preprocessed pairs")
    print(f"   Resuming from batch {last_batch + 1}")
else:
    print("\n2. Starting fresh preprocessing...")
    train_entries = []
    last_batch = -1
    transformer = None

# Preprocessing with checkpoints
if len(train_entries) < TRAIN_SIZE:
    print("\n3. Preprocessing training pairs (with checkpoints)...")

    start_idx = len(train_entries)

    for i in range(start_idx, len(train_data)):
        ex = train_data[i]
        entry1 = prepare_entry(ex['text1'], mode='fast')
        entry2 = prepare_entry(ex['text2'], mode='fast')
        train_entries.append((entry1, entry2, ex['same']))

        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(train_data)}", end='\r')
            sys.stdout.flush()

        # Save checkpoint every BATCH_SIZE
        if (i + 1) % BATCH_SIZE == 0:
            batch_num = i // BATCH_SIZE
            print(f"\n   ✓ Checkpoint at {i+1} pairs (batch {batch_num})")

            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'train_entries': train_entries,
                    'last_batch': batch_num,
                    'transformer': transformer
                }, f)

            print(f"   Saved checkpoint to {checkpoint_file}")

    print(f"\n   ✓ Preprocessing complete: {len(train_entries)} pairs")
else:
    print("\n3. ✓ Preprocessing already complete")

# Fit transformer (if not already done)
transformer_file = os.path.join(CHECKPOINT_DIR, 'transformer.pkl')

if os.path.exists(transformer_file):
    print("\n4. Loading saved transformer...")
    with open(transformer_file, 'rb') as f:
        transformer = pickle.load(f)
    print("   ✓ Transformer loaded")
else:
    print("\n4. Fitting feature transformer...")
    all_train_texts = []
    for e1, e2, label in train_entries:
        all_train_texts.append(e1)
        all_train_texts.append(e2)

    transformer = get_transformer()
    transformer.fit(all_train_texts)

    with open(transformer_file, 'wb') as f:
        pickle.dump(transformer, f)

    print(f"   ✓ Transformer saved to {transformer_file}")

# Extract features (with checkpoints)
features_file = os.path.join(CHECKPOINT_DIR, 'features.pkl')

if os.path.exists(features_file):
    print("\n5. Loading extracted features...")
    with open(features_file, 'rb') as f:
        feature_data = pickle.load(f)

    train_features = feature_data['train_features']
    print(f"   ✓ Loaded {len(train_features)} feature vectors")
else:
    print("\n5. Extracting feature vectors (this may take a while)...")
    train_features = []

    for i, (e1, e2, label) in enumerate(train_entries):
        f1 = transformer.transform([e1]).toarray()[0]
        f2 = transformer.transform([e2]).toarray()[0]
        train_features.append((f1, f2, label))

        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(train_entries)}", end='\r')
            sys.stdout.flush()

    print(f"\n   ✓ Feature extraction complete")

    # Save features
    with open(features_file, 'wb') as f:
        pickle.dump({'train_features': train_features}, f)

    print(f"   Saved features to {features_file}")

print(f"\n   Feature vector size: {len(train_features[0][0])}")

# Standardize and compute differences
scaler_file = os.path.join(CHECKPOINT_DIR, 'scalers.pkl')

if os.path.exists(scaler_file):
    print("\n6. Loading saved scalers...")
    with open(scaler_file, 'rb') as f:
        scaler_data = pickle.load(f)

    feature_scaler = scaler_data['feature_scaler']
    X_train = scaler_data['X_train']
    y_train = scaler_data['y_train']

    print(f"   ✓ Loaded training matrix: {X_train.shape}")
else:
    print("\n6. Standardizing features and computing differences...")

    # Standardize features
    all_features = []
    for f1, f2, label in train_features:
        all_features.append(f1)
        all_features.append(f2)
    all_features = np.array(all_features)

    feature_scaler = StandardScaler()
    feature_scaler.fit(all_features)

    print("   Computing differences...")
    X_train_list = []
    y_train = []

    for i, (f1, f2, label) in enumerate(train_features):
        f1_std = feature_scaler.transform([f1])[0]
        f2_std = feature_scaler.transform([f2])[0]

        # Use both absolute difference and product
        abs_diff = np.abs(f1_std - f2_std)
        product = f1_std * f2_std
        combined = np.concatenate([abs_diff, product])

        X_train_list.append(combined)
        y_train.append(label)

        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(train_features)}", end='\r')
            sys.stdout.flush()

    X_train_raw = np.array(X_train_list)
    y_train = np.array(y_train)

    print(f"\n   Combined feature size: {X_train_raw.shape[1]}")

    # Feature selection
    print("\n7. Feature selection...")
    var_threshold = VarianceThreshold(threshold=0.01)
    X_train_var = var_threshold.fit_transform(X_train_raw)
    print(f"   After variance threshold: {X_train_var.shape[1]} features")

    k = min(2000, X_train_var.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_var, y_train)
    print(f"   After selecting top {k}: {X_train_selected.shape[1]} features")

    # Final standardization
    diff_scaler = StandardScaler()
    X_train = diff_scaler.fit_transform(X_train_selected)

    print(f"   Final training matrix: {X_train.shape}")

    # Save everything
    with open(scaler_file, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'var_threshold': var_threshold,
            'selector': selector,
            'diff_scaler': diff_scaler,
            'X_train': X_train,
            'y_train': y_train
        }, f)

    print(f"   ✓ Saved scalers and training matrix")

# Train classifier (LinearSVC - much faster than RBF SVM)
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

    print(f"   Saved classifier to {classifier_file}")

# Load all components for evaluation
print("\n9. Loading all model components for evaluation...")

with open(scaler_file, 'rb') as f:
    scaler_data = pickle.load(f)

feature_scaler = scaler_data['feature_scaler']
var_threshold = scaler_data['var_threshold']
selector = scaler_data['selector']
diff_scaler = scaler_data['diff_scaler']

print("   ✓ All components loaded")

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
print("\n10. Evaluating on validation set...")
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
print("\n11. Testing on test set...")
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
print("FINAL RESULTS: CHECKPOINTED FEATURE VECTOR MODEL")
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
print(f"Checkpointed Feature Vector:     Acc={accuracy*100:.1f}%, F1={f1*100:.1f}%, AUC={auc:.3f}")
print()
print("Fine-tuned BERT:                 Acc=70.1%, F1=71.6%, AUC=0.760")

# Save final model for ensemble
print("\n12. Saving final model for ensemble...")
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
