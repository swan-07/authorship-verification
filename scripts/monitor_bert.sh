#!/bin/bash
# Monitor BERT download/evaluation progress

echo "Monitoring BERT evaluation..."
echo "Press Ctrl+C to stop monitoring"
echo ""

OUTPUT_FILE="/tmp/claude/-Users-swan-Documents-GitHub-authorship-verification/tasks/b536916.output"

while true; do
    clear
    echo "==================================================================="
    echo "BERT Evaluation Monitor - $(date '+%H:%M:%S')"
    echo "==================================================================="
    echo ""

    # Check if process is still running
    if ps aux | grep "step3_test_bert" | grep -v grep > /dev/null; then
        echo "Status: ✓ RUNNING"
        echo ""

        # Show last 30 lines of output
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Latest output:"
            echo "-------------------------------------------------------------------"
            tail -30 "$OUTPUT_FILE" 2>/dev/null
            echo "-------------------------------------------------------------------"
        else
            echo "No output file yet..."
        fi
    else
        echo "Status: ✓ COMPLETED (or stopped)"
        echo ""
        echo "Final output:"
        echo "-------------------------------------------------------------------"
        if [ -f "$OUTPUT_FILE" ]; then
            cat "$OUTPUT_FILE"
        else
            echo "No output file found"
        fi
        echo "==================================================================="
        break
    fi

    echo ""
    echo "Refreshing in 10 seconds... (Ctrl+C to stop)"
    sleep 10
done

echo ""
echo "Done! Check the full results in:"
echo "  $OUTPUT_FILE"
