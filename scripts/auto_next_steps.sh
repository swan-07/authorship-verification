#!/bin/bash

# This script will automatically run the next steps once feature vector training completes

echo "Waiting for feature vector training to complete..."

# Wait for the process to finish
while ps aux | grep -q "[t]rain_feature_vector_checkpointed.py"; do
    sleep 60  # Check every minute
done

echo "Feature vector training complete!"
echo "Starting ensemble creation..."

# Run ensemble script (will be created)
python3 create_ensemble.py 2>&1 | tee ensemble_training.log

echo "Ensemble complete! Check ensemble_training.log for results"
echo ""
echo "Next step: Run python3 finetune_bert_full.py to train BERT on all 325K pairs"
