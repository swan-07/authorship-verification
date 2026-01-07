#!/bin/bash
LOG_FILE="bert_100k_bs4.log"
echo "Starting BERT training monitor..."
echo "Log file: $LOG_FILE"
echo ""

while true; do
    clear
    echo "========================================"
    echo "BERT TRAINING MONITOR"
    echo "========================================"
    date
    echo ""
    
    # Check if process is running
    if ps aux | grep -q "[f]inetune_bert_full.py"; then
        echo "Status: TRAINING ACTIVE ✓"
        
        # Get latest progress
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Latest progress:"
            tail -3 "$LOG_FILE" | grep -E "it/s|Checkpoint saved"
            
            echo ""
            echo "Recent lines:"
            tail -10 "$LOG_FILE"
            
            # Check for checkpoints
            echo ""
            echo "Checkpoints saved:"
            ls -lh bert_full_checkpoints/ 2>/dev/null | grep checkpoint || echo "  None yet"
        fi
    else
        echo "Status: NOT RUNNING"
        echo ""
        echo "Last 20 lines of log:"
        tail -20 "$LOG_FILE" 2>/dev/null || echo "Log file not found"
        echo ""
        echo "Training appears to have finished or crashed."
        break
    fi
    
    echo ""
    echo "========================================"
    echo "Next update in 5 minutes..."
    echo "Press Ctrl+C to stop monitoring"
    sleep 300
done
