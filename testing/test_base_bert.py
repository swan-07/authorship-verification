import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

print("=" * 80)
print("TESTING BASE BERT ON ALL TEST DATASETS")
print("=" * 80)

start_time = time.time()
last_update = start_time

model = SentenceTransformer('bert-base-cased')
dataset = load_dataset('swan07/authorship-verification')

# Get test set
test = dataset['test']
print(f"\nTotal test samples: {len(test)}")

# Compute all scores
print("\nComputing cosine similarity scores...")
scores, labels = [], []
for i, ex in enumerate(test):
    e1, e2 = model.encode(ex['text1']), model.encode(ex['text2'])
    scores.append(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    labels.append(ex['same'])

    # Progress update every 1000 samples or 5 minutes
    if i % 1000 == 0 or (time.time() - last_update) > 300:
        elapsed = (time.time() - start_time) / 60
        print(f"  Progress: {i}/{len(test)} ({i/len(test)*100:.1f}%) - Elapsed: {elapsed:.1f} min")
        last_update = time.time()

scores = np.array(scores)
labels = np.array(labels)

print("\n" + "=" * 80)
print("BASE BERT RESULTS (threshold=0.5)")
print("=" * 80)

preds = (scores >= 0.5).astype(int)
print(f"Accuracy:  {accuracy_score(labels, preds):.1%}")
print(f"Precision: {precision_score(labels, preds):.1%}")
print(f"Recall:    {recall_score(labels, preds):.1%}")
print(f"F1 Score:  {f1_score(labels, preds):.1%}")

# Save scores for later use
np.savez('test_scores.npz', scores=scores, labels=labels)
print("\nScores saved to test_scores.npz")
print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
