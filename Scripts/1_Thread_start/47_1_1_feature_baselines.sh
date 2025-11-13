#!/bin/bash

# Usage: bash run_all_models.sh <run_number> <max_feats> <stage1_model>
if [ $# -lt 3 ]; then
    echo "Usage: bash 47_1_1_feature_baselines.sh <run_number> <collection> <calibration> <max_feats>"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: $0 $@"

RUN_NUMBER=$1
COLLECTION=$2
CALIBRATION=$3
MAX_FEATS=$4
SUBREDDITS=("crypto" "conspiracy" "politics")

for SUB in "${SUBREDDITS[@]}"; do
    echo "======================================"
    echo "▶ Running Stage 1.1 Feature Baselines for $SUB"
    echo "======================================"

    # Set LOGDIR based on COLLECTION value
    if [ "$COLLECTION" -eq 0 ]; then
        LOGDIR="REGDATA/outputs/${SUB}/Run_${RUN_NUMBER}/logs"
    else
        LOGDIR="REGDATA/outputs/${SUB}_14/Run_${RUN_NUMBER}/logs"
    fi
    mkdir -p "$LOGDIR"

    # Stage 1 Tuning
    echo "▶ Running ($SUB)..."
    python -u 47_STAGE_1_1_FEATURE_BASELINE.py "$SUB" "$RUN_NUMBER" "$COLLECTION" "$CALIBRATION" "$MAX_FEATS" > "$LOGDIR/1_1_feature_baselines_$CALIBRATION.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Baseliens failed for $SUB. Skipping next steps."
        continue
    fi

    echo "✅ $SUB pipeline part 1.1 completed successfully."
    echo ""
done

echo "✅ All pipelines completed."

