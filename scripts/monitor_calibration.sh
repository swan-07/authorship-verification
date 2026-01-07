#!/bin/bash
# Monitor calibration training progress

OUTPUT_FILE="/tmp/claude/-Users-swan-Documents-GitHub-authorship-verification/tasks/b3ac2f1.output"
RESULT_FILE="/Users/swan/Documents/GitHub/authorship-verification/calibration_model_base_bert.pkl"

echo "Monitoring calibration training..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Check if result file exists (training complete)
    if [ -f "$RESULT_FILE" ]; then
        echo "✓ CALIBRATION COMPLETE!"
        echo ""
        echo "Results:"
        cat "$OUTPUT_FILE" | tail -50
        break
    fi

    # Check if process is still running
    if ! ps aux | grep "step6_train_calibration" | grep -v grep > /dev/null; then
        echo "⚠ Process stopped"
        echo "Last output:"
        tail -30 "$OUTPUT_FILE"
        break
    fi

    # Show progress
    clear
    echo "Calibration Training in Progress..."
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    echo "Latest output:"
    echo "-------------------------------------------------------------------"
    tail -20 "$OUTPUT_FILE" 2>/dev/null
    echo "-------------------------------------------------------------------"

    sleep 10
done
