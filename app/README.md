# Authorship Verification App

Interactive web application for authorship verification using BERT + Stylometric analysis.

## Features

- **Three Models**: Compare predictions from BERT, Stylometric, and Ensemble models
- **Interpretability**: Detailed feature breakdowns and visualizations
- **Real Examples**: Load examples from the test dataset
- **Interactive**: Paste any two texts and get instant predictions

## Running the App

```bash
cd app
streamlit run authorship_app.py
```

The app will open at http://localhost:8501

## Models Used

- **BERT**: Fine-tuned sentence-transformers model (73.9% accuracy)
- **Stylometric**: Character n-grams, word patterns, punctuation (62.2% accuracy)
- **Ensemble**: Logistic regression combining both (73.9% accuracy, 0.823 AUC)

## Dataset

Models trained on [swan07/authorship-verification](https://huggingface.co/datasets/swan07/authorship-verification) (325K text pairs from 12 sources including PAN competitions, Reuters, arXiv, etc.)
