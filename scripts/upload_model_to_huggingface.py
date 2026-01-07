#!/usr/bin/env python3
"""
Upload BERT model to HuggingFace Model Hub for Streamlit deployment

Usage:
    python upload_model_to_huggingface.py

Requirements:
    pip install huggingface-hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "bert-finetuned-authorship"
REPO_ID = "swan07/bert-authorship-verification"  # Change to your HF username

def upload_model():
    """Upload BERT model to HuggingFace"""

    print("=" * 60)
    print("Uploading BERT Model to HuggingFace")
    print("=" * 60)

    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Make sure you've trained the model first.")
        return

    print(f"\n📦 Model path: {MODEL_PATH}")
    print(f"🎯 Target repo: {REPO_ID}")

    # Check for authentication
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"\n✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print("\n❌ Not logged in to HuggingFace!")
        print("\nPlease login first:")
        print("  huggingface-cli login")
        print("\nOr set your token:")
        print("  export HUGGING_FACE_HUB_TOKEN=your_token_here")
        return

    # Create repository
    try:
        print(f"\n📝 Creating repository: {REPO_ID}")
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✓ Repository created/verified")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")

    # Upload model files
    print("\n⬆️  Uploading model files...")
    try:
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("✓ Model uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return

    # Create README
    readme_content = f"""---
language: en
license: mit
tags:
- authorship-verification
- sentence-transformers
- sentence-similarity
datasets:
- swan07/authorship-verification
metrics:
- accuracy
- auc
model-index:
- name: {REPO_ID}
  results:
  - task:
      type: authorship-verification
      name: Authorship Verification
    dataset:
      name: swan07/authorship-verification
      type: authorship-verification
    metrics:
    - type: accuracy
      value: 0.739
      name: Accuracy
    - type: auc
      value: 0.821
      name: AUC
---

# BERT for Authorship Verification

Fine-tuned BERT model for determining if two texts were written by the same author.

## Model Details

- **Base Model**: sentence-transformers/all-MiniLM-L6-v2
- **Training Data**: 50K text pairs from swan07/authorship-verification dataset
- **Task**: Authorship verification (binary classification)
- **Performance**: 73.9% accuracy, 0.821 AUC

## Usage

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer('{REPO_ID}')

# Encode texts
text1 = "Your first text here"
text2 = "Your second text here"

emb1 = model.encode(text1)
emb2 = model.encode(text2)

# Calculate cosine similarity
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Predict
prediction = "Same Author" if similarity >= 0.5 else "Different Authors"
print(f"Prediction: {{prediction}}")
print(f"Similarity: {{similarity:.3f}}")
```

## Training

Trained on 50K pairs from the swan07/authorship-verification dataset using:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 4
- Loss: CosineSimilarityLoss

## Dataset

[swan07/authorship-verification](https://huggingface.co/datasets/swan07/authorship-verification) - 325K text pairs from 12 sources including PAN competitions (2011-2020).

## Citation

```bibtex
@article{{manolache2021transferring,
  title={{Transferring BERT-like Transformers' Knowledge for Authorship Verification}},
  author={{Manolache, Andrei and Brad, Florin and Burceanu, Elena and Barbalau, Antonio and Ionescu, Radu Tudor and Popescu, Marius}},
  journal={{arXiv preprint arXiv:2112.05125}},
  year={{2021}}
}}
```

## Links

- **Live Demo**: [same-writer-detector.streamlit.app](https://same-writer-detector.streamlit.app/)
- **Code**: [github.com/swan-07/authorship-verification](https://github.com/swan-07/authorship-verification)
- **Dataset**: [huggingface.co/datasets/swan07/authorship-verification](https://huggingface.co/datasets/swan07/authorship-verification)
"""

    try:
        print("\n📄 Creating model card (README.md)...")
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("✓ Model card created")
    except Exception as e:
        print(f"⚠️  Could not create model card: {e}")

    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print(f"\n🔗 Model URL: https://huggingface.co/{REPO_ID}")
    print("\n📝 Next steps:")
    print("1. Update your Streamlit app to use:")
    print(f"   SentenceTransformer('{REPO_ID}')")
    print("2. Deploy to streamlit-av repository")
    print("3. Push changes to GitHub")
    print("\nStreamlit Cloud will auto-deploy with the new model!")

if __name__ == "__main__":
    upload_model()
