import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

print("=" * 80)
print("TESTING BASE BERT (USING CACHED DATASET)")
print("=" * 80)

start = time.time()

# Load cached dataset (instant!)
print("\nLoading cached dataset...")
with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)
print(f"  Loaded in {time.time()-start:.1f}s")

print("\nLoading BERT model...")
model = SentenceTransformer('bert-base-cased')

test = dataset['test']
print(f"\nComputing scores on {len(test)} samples...")

scores, labels = [], []
for i, ex in enumerate(test):
    if i % 1000 == 0:
        print(f"  {i}/{len(test)} ({i/len(test)*100:.1f}%)")
    e1, e2 = model.encode(ex['text1']), model.encode(ex['text2'])
    scores.append(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    labels.append(ex['same'])

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

np.savez('test_scores_cached.npz', scores=scores, labels=labels)
print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
