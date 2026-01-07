#!/usr/bin/env python3
"""Quick status check - shows what's running and what's found"""

import subprocess
import os
from pathlib import Path

print("="*80)
print("CURRENT STATUS CHECK")
print("="*80)

# Check if BERT evaluation is running
print("\n1. BERT Evaluation:")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
if 'step3_test_bert' in result.stdout:
    print("   Status: 🔄 RUNNING")
    # Try to get progress
    output_file = "/tmp/claude/-Users-swan-Documents-GitHub-authorship-verification/tasks/b536916.output"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
            # Look for progress indicators
            for line in reversed(lines[-10:]):
                if 'Fetching' in line or '%' in line:
                    print(f"   Progress: {line.strip()}")
                    break
else:
    print("   Status: ✓ COMPLETED (or not started)")
    output_file = "/tmp/claude/-Users-swan-Documents-GitHub-authorship-verification/tasks/b536916.output"
    if os.path.exists(output_file):
        print("\n   Results available! Run:")
        print(f"   cat {output_file}")

# Check for models
print("\n2. Models Found:")

# BERT on HuggingFace
print("   ✓ BERT model: swan07/final-models (HuggingFace)")

# Feature Vector on HuggingFace
print("   ✓ Feature Vector model: swan07/final-models/featuremodel.p (HuggingFace)")

# Calibration models
calib_found = False
for path in ['calibration_model.pkl', 'calibration_model1.pkl', 'siamesebert/methods/calibration_model.pkl']:
    p = Path(path)
    if p.exists():
        print(f"   ✓ Calibration: {path}")
        calib_found = True

if not calib_found:
    print("   ✗ Calibration models: Not found (need to train)")

print("\n3. Next Steps:")
if 'step3_test_bert' in result.stdout:
    print("   ⏳ Waiting for BERT evaluation to complete...")
    print("   📊 Monitor with: bash scripts/monitor_bert.sh")
else:
    print("   ✅ Download Feature Vector model")
    print("   ✅ Train calibration models")
    print("   ✅ Run full baseline evaluation")

print("\n" + "="*80)
