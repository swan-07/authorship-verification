# Deploying to Streamlit Cloud

This guide explains how to deploy your updated models to the live app at https://same-writer-detector.streamlit.app/

## 🎯 Quick Summary

The BERT model (413MB) is too large for GitHub. We'll upload it to HuggingFace and let Streamlit download it automatically.

## 📋 Step-by-Step Guide

### 1. Upload BERT Model to HuggingFace

```bash
# Install HuggingFace CLI (if needed)
pip install huggingface-hub

# Login to HuggingFace (one-time setup)
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens

# Upload model (automated script)
python scripts/upload_model_to_huggingface.py
```

This uploads your model to: `swan07/bert-authorship-verification`

### 2. Prepare Files for Deployment

```bash
# Create deployment folder
mkdir -p ~/deploy-streamlit
cd ~/deploy-streamlit

# Clone your streamlit-av repo
git clone https://github.com/swan-07/streamlit-av
cd streamlit-av

# Copy updated app
cp ~/Documents/GitHub/authorship-verification/app/authorship_app_huggingface.py ./authorship_app.py

# Copy small models (these are fine to commit)
cp ~/Documents/GitHub/authorship-verification/models/stylometric_pan_model.pkl .
cp ~/Documents/GitHub/authorship-verification/models/ensemble_model.pkl .
```

### 3. Update Requirements

Create/update `requirements.txt`:

```txt
streamlit
sentence-transformers
scikit-learn
numpy
plotly
```

### 4. Test Locally

```bash
streamlit run authorship_app.py
```

Visit http://localhost:8501 and verify:
- ✓ Models load successfully
- ✓ Example buttons work
- ✓ Analysis produces results
- ✓ Visualizations display correctly

### 5. Deploy to GitHub

```bash
git add .
git commit -m "Update to new models (73.9% accuracy, +3.8% improvement)

- Load BERT from HuggingFace (swan07/bert-authorship-verification)
- Update to ensemble model with stylometric features
- Improve accuracy from 70.1% to 73.9%
- Improve AUC from 0.760 to 0.823
- Add real examples from test dataset
- Enhanced interpretability features"

git push origin main
```

### 6. Streamlit Cloud Auto-Deploys

Streamlit Cloud detects the push and:
1. Downloads BERT model from HuggingFace (cached after first download)
2. Loads .pkl files from repo
3. Rebuilds and deploys automatically

**Timeline**: Usually takes 2-5 minutes

Check deployment at: https://same-writer-detector.streamlit.app/

## 🔍 Troubleshooting

### Model Download Issues

If Streamlit can't download the model:

1. **Check model is public**: Visit https://huggingface.co/swan07/bert-authorship-verification
2. **Check Streamlit logs**: In Streamlit Cloud dashboard
3. **Manually trigger rebuild**: Streamlit Cloud dashboard → Reboot app

### Out of Memory

If Streamlit runs out of memory:

1. The BERT model uses ~500MB RAM
2. Streamlit free tier: 1GB RAM
3. Upgrade to Streamlit Cloud paid tier if needed

### File Not Found

If `.pkl` files aren't found:

1. Check they're committed to git: `git ls-files | grep pkl`
2. Check file paths match in `authorship_app.py`

## 📊 Performance After Deployment

Your live app will have:
- **Accuracy**: 73.9% (+3.8% from 70.1%)
- **AUC**: 0.823 (+0.063 from 0.760)
- **Training Data**: 50K pairs
- **Dataset**: 325K pairs from 12 sources

## 🔄 Future Updates

To update the model later:

1. Train new model locally
2. Upload to HuggingFace: `python scripts/upload_model_to_huggingface.py`
3. Update version in app if needed
4. Push to GitHub
5. Streamlit auto-deploys

## 📚 Additional Resources

- **HuggingFace Models**: https://huggingface.co/models
- **Streamlit Cloud**: https://streamlit.io/cloud
- **Sentence Transformers**: https://www.sbert.net/

---

**Questions?** See the main [README.md](README.md) or open an issue on GitHub.
