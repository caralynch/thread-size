#!/bin/bash

# Usage: bash run_all_models.sh <run_number> <max_feats> <stage1_model>
if [ $# -lt 3 ]; then
    echo "Usage: bash 47_2_1_pipeline.sh <run_number> <collection> <calibration> <n_classes> <max_feats>"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: $0 $@"

RUN_NUMBER=$1
COLLECTION=$2
CALIBRATION=$3
N_CLASSES=$4
MAX_FEATS=$5

# Arrays to store parsed lines
SUBREDDIT_LIST=("crypto" "conspiracy" "politics")

for i in "${!SUBREDDIT_LIST[@]}"; do
    SUB="${SUBREDDIT_LIST[$i]}"
    echo "======================================"
    echo "▶ Running Stage 2.1 Feature Baselines for $SUB."
    echo "======================================"

    # Set LOGDIR based on COLLECTION value
    if [ "$COLLECTION" -eq 0 ]; then
        LOGDIR="REGDATA/outputs/${SUB}/Run_${RUN_NUMBER}/logs"
    else
        LOGDIR="REGDATA/outputs/${SUB}_14/TwoStageModel/Run_${RUN_NUMBER}/logs"
    fi
    mkdir -p "$LOGDIR"

    echo "Feature baselines"
    python -u 49_1_FEATURE_BASELINE.py "$SUB" "$RUN_NUMBER" "$COLLECTION" "$CALIBRATION" "$N_CLASSES" "$MAX_FEATS"> "$LOGDIR/1_feature_baseline.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Baselines failed for $SUB. Skipping next steps."
        continue
    fi
done

echo "Done"