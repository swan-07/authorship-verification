# Directory Structure

This document describes the organization of the authorship verification project.

## Main Directories

### `/app`
- **Purpose**: Streamlit web application
- **Contents**: `authorship_app.py` - Interactive demo of the authorship verification system

### `/models`
- **Purpose**: Trained models and model artifacts
- **Contents**:
  - `bert-finetuned-authorship/` - Fine-tuned BERT model (73.9% accuracy)
  - `stylometric_pan_model.pkl` - Stylometric classifier (62.2% accuracy)
  - `ensemble_model.pkl` - Ensemble combiner (73.9% accuracy, 0.823 AUC)
  - Calibration models and thresholds

### `/training`
- **Purpose**: Scripts for training models
- **Contents**:
  - `train_stylometric_pan.py` - Train stylometric model with PAN-style features
  - `finetune_bert*.py` - Fine-tune BERT for authorship verification
  - `create_ensemble.py` - Create ensemble combining BERT + Stylometric
  - `train_feature_vector*.py` - Various feature vector approaches

### `/testing`
- **Purpose**: Scripts for evaluating models
- **Contents**:
  - `test_ensemble_detailed.py` - Detailed ensemble evaluation
  - `test_finetuned*.py` - Test fine-tuned BERT models
  - `test_*.py` - Various model testing scripts

### `/analysis`
- **Purpose**: Analysis and visualization scripts
- **Contents**:
  - `visualize_*.py` - Generate visualizations
  - `interpret_bert.py` - BERT interpretability analysis
  - `probe_dimensions.py` - Dimension probing experiments
  - `analyze_*.py` - Error analysis and model analysis

### `/logs`
- **Purpose**: Training and testing logs
- **Contents**: All `.log` files from training runs

### `/visualizations`
- **Purpose**: Generated plots and figures
- **Contents**: All `.png` and `.pdf` visualization files

### `/data`
- **Purpose**: Cached data and test results
- **Contents**: `.npz` files, cached datasets, preprocessed data

### `/checkpoints_archive`
- **Purpose**: Training checkpoints (archived)
- **Contents**: Model checkpoints from interrupted training runs

### `/scripts`
- **Purpose**: Utility scripts
- **Contents**: Helper scripts, shell scripts, setup scripts

### `/docs`
- **Purpose**: Documentation
- **Contents**: 
  - `transcript.md` - Development transcript
  - `process.MD` - Process documentation

### `/archive`
- **Purpose**: Old/deprecated code
- **Contents**: Jupyter notebooks, old experiments

### `/featurevector`
- **Purpose**: Feature vector baseline code (from research paper)

### `/siamesebert`
- **Purpose**: Siamese BERT baseline code (from research paper)

## Key Files in Root

- `README.md` - Project overview and instructions
- `requirements.txt` - Python dependencies
- `LICENSE` - Project license
- `DIRECTORY_STRUCTURE.md` - This file

## Running the App

```bash
cd app
streamlit run authorship_app.py
```

## Training Models

```bash
# Train stylometric model
python training/train_stylometric_pan.py

# Fine-tune BERT
python training/finetune_bert_v2.py

# Create ensemble
python training/create_ensemble.py
```

## Testing Models

```bash
# Test ensemble
python testing/test_ensemble_detailed.py
```
