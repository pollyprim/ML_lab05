#!/usr/bin/env bash
# Full reproduction script for the Week-5 satellite classification lab.
# Usage: DATA_DIR=/path/to/competition/data ./run_all.sh
# Required files in DATA_DIR: train.csv, val.csv, test.csv and folders train/, val/, test/
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-./data}"

echo "=== Experiment 1: MLP (64px) ==="
DATA_DIR=$DATA_DIR python3 compare.py mlp
echo "=== Experiment 2: simple CNN (64px) ==="
DATA_DIR=$DATA_DIR python3 compare.py cnn
echo "=== Experiment 3: advanced CNN (64px) ==="
DATA_DIR=$DATA_DIR python3 compare.py advcnn
echo "=== Experiment 4: ResNet-18 pretrained from scratch head+layer4 (64px) ==="
DATA_DIR=$DATA_DIR python3 compare.py resnet
echo "=== Experiment 5: artifact diagnostics on val/test ==="
python3 check_artifacts.py "$DATA_DIR" || true
echo "=== Experiment 6: final model ablations (128px, robust aug) ==="
DATA_DIR=$DATA_DIR python3 final_experiment.py train 25
DATA_DIR=$DATA_DIR python3 final_experiment.py finetune 25
DATA_DIR=$DATA_DIR python3 final_experiment.py scratch 40
echo "=== Final pipeline: train on train+val, predict test -> final_submission.csv ==="
python3 final_submission.py "$DATA_DIR" final_submission.csv 128
echo "ALL DONE"
