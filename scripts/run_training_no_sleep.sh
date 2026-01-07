#!/bin/bash
# Run training with caffeinate to prevent sleep

echo "=================================="
echo "STARTING TRAINING (NO SLEEP MODE)"
echo "=================================="
echo ""
echo "This will prevent your Mac from sleeping during training."
echo "Training will run in the background."
echo ""

# Kill any existing training processes
pkill -f "finetune_bert.py"
pkill -f "train_calibration"

# Clean old logs
rm -f finetune_bert.log calibration_quick.log

echo "1. Starting calibration training (10-15 min)..."
caffeinate -i python3 scripts/step6_train_calibration.py > calibration_quick.log 2>&1 &
CAL_PID=$!
echo "   Calibration PID: $CAL_PID"
echo "   Monitor: tail -f calibration_quick.log"

echo ""
echo "2. Starting BERT fine-tuning (2-4 hours)..."
caffeinate -i python3 scripts/finetune_bert.py > finetune_bert.log 2>&1 &
BERT_PID=$!
echo "   BERT PID: $BERT_PID"
echo "   Monitor: tail -f finetune_bert.log"

echo ""
echo "=================================="
echo "BOTH JOBS RUNNING IN BACKGROUND"
echo "=================================="
echo ""
echo "Your computer will NOT sleep while training runs."
echo "You can close this terminal - training continues."
echo ""
echo "Check status:"
echo "  ps aux | grep -E '(finetune_bert|train_calibration)' | grep -v grep"
echo ""
echo "Monitor progress:"
echo "  tail -f finetune_bert.log"
echo "  tail -f calibration_quick.log"
echo ""
echo "When done, models will be saved to:"
echo "  - calibration_model_base_bert.pkl"
echo "  - ./finetuned_bert_model/"
echo ""
