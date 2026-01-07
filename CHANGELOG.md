# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - January 2026

### Added
- **Stylometric Model**: New PAN competition-style character n-gram model
  - Character n-grams (3, 4, 5)
  - Word n-grams
  - Punctuation pattern analysis
  - Text length and sentence structure features
  - Achieves 62.2% accuracy, 0.665 AUC

- **Ensemble Model**: Logistic regression combining BERT + Stylometric
  - Achieves 73.9% accuracy, 0.823 AUC
  - BERT weighted ~7x more than stylometric
  - Trained on validation set predictions
  - **+3.8% accuracy improvement over previous best (70.1% → 73.9%)**
  - **+0.063 AUC improvement (0.760 → 0.823)**

- **Improved BERT Training**: Streamlined approach using sentence-transformers
  - Fine-tuned on 50K pairs
  - Achieves 73.9% accuracy, 0.821 AUC
  - Faster training (2 hours on A100 vs many hours for original)
  - **+3.8% accuracy improvement over previous BERT approach (70.1% → 73.9%)**

- **Local Streamlit App**: Interactive demo in `app/` folder
  - Three model comparison (BERT, Stylometric, Ensemble)
  - Full interpretability features
  - Real examples from test dataset
  - Detailed feature breakdowns and visualizations

- **Repository Organization**: Clean directory structure
  - `app/` - Streamlit application
  - `models/` - Trained models
  - `training/` - Training scripts
  - `testing/` - Evaluation scripts
  - `analysis/` - Visualization and interpretability
  - `logs/` - Training logs
  - `visualizations/` - Generated plots
  - `data/` - Cached data

- **Documentation**:
  - Comprehensive README with badges and clear sections
  - DIRECTORY_STRUCTURE.md for repository navigation
  - app/README.md for local demo instructions
  - .gitignore for proper git management

### Changed
- Reorganized 100+ files from root directory into logical folders
- Updated README with modern formatting and clear results table
- Improved model paths to use relative references

### Performance Comparison

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| **v1.0 (Original)** | | | |
| Base BERT + Calibration | 63.7% | 58.2% | 0.676 |
| Fine-tuned BERT + Calibration | 70.1% | 71.6% | 0.760 |
| Feature Vector | 58.6% | 57.9% | 0.619 |
| **v2.0 (New)** | | | |
| BERT (fine-tuned, simplified) | 73.9% | 73.8% | 0.821 |
| Stylometric (PAN-style) | 62.2% | 57.1% | 0.665 |
| **Ensemble (BERT + Stylometric)** | **73.9%** | **73.8%** | **0.823** |

**Improvements:**
- +3.8% accuracy (70.1% → 73.9%)
- +2.2 F1 points (71.6% → 73.8%)
- +0.063 AUC (0.760 → 0.823)

### Technical Details
- Training scripts use `sentence-transformers` library
- Models saved in `models/` directory
- All log files in `logs/` directory
- Visualizations in `visualizations/` directory
- Test data cached in `data/` directory

## [1.0.0] - Original Release

### Original Work (From Research Paper)

**Feature Vector Model** (`featurevector/`):
- Modified from [pan2021_authorship_verification](https://github.com/janithnw/pan2021_authorship_verification)
- Deep linguistic feature extraction
- Preprocessing via Jupyter notebooks
- Training and prediction notebooks

**Siamese BERT Model** (`siamesebert/`):
- Based on [Valla](https://github.com/JacobTyo/Valla)
- Embedding-based approach
- Logistic regression calibration
- Attention-based highlighting

**Dataset Curation**:
- 325K text pairs from 12 sources
- Uploaded to [HuggingFace](https://huggingface.co/datasets/swan07/authorship-verification)
- Includes PAN 2011-2020, Reuters50, IMDB62, Blog Corpus, arXiv, Victorian Era, BAWE, DarkReddit
- Curation code in `archive/Authorship_Verification_Datasets.ipynb`

**Research Paper**:
- "Transparent Authorship Verification"
- Available at https://swan-07.github.io/assets/Transparent%20Authorship%20Verification.pdf

**Demo**:
- Separate repository: https://github.com/swan-07/streamlit-av
- Live at: https://same-writer-detector.streamlit.app/

---

## Comparison

### What Changed?

**Simplification**: New models use simpler, more maintainable approaches
- Old: Complex feature extraction pipelines
- New: Simple n-grams + sentence-transformers

**Organization**: Better repository structure
- Old: 100+ files in root directory
- New: Clean folder hierarchy

**Documentation**: Improved clarity
- Old: Jupyter notebooks with inline documentation
- New: Detailed README, DIRECTORY_STRUCTURE.md, Python scripts

**Performance**: Competitive accuracy with faster training
- Old: Multiple hours for feature extraction + training
- New: 2-3 hours total training time

**Demo**: Local app available
- Old: Separate streamlit repository only
- New: Local `app/` folder + separate hosted version

### What Stayed the Same?

- **Dataset**: Same 325K pairs from 12 sources
- **Original Models**: Feature Vector and Siamese BERT code preserved in `featurevector/` and `siamesebert/`
- **Research Paper**: Original research and findings remain the foundation
- **Live Demo**: Production app still hosted separately

Both approaches are valuable:
- **Original models** (paper): More sophisticated, research-grade
- **New models** (2026): Simpler, faster, easier to reproduce
