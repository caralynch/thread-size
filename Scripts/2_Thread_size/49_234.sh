#!/bin/bash

# Usage: bash run_all_models.sh <run_number> <max_feats> <stage1_model>
if [ $# -lt 3 ]; then
    echo "Usage: bash 47_2_pipeline.sh <run_number> <collection> <calibration> <n_classes>"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: $0 $@"

RUN_NUMBER=$1
COLLECTION=$2
CALIBRATION=$3
N_CLASSES=$4

# Arrays to store parsed lines
SUBREDDIT_LIST=("crypto" "conspiracy" "politics")


for i in "${!SUBREDDIT_LIST[@]}"; do
    SUB="${SUBREDDIT_LIST[$i]}"
    # echo "======================================"
    # echo "▶ Running Stage 2.1 Feature Baselines for $SUB"
    # echo "======================================"

    # Set LOGDIR based on COLLECTION value
    if [ "$COLLECTION" -eq 0 ]; then
        LOGDIR="REGDATA/outputs/${SUB}/Run_${RUN_NUMBER}/logs"
    else
        LOGDIR="REGDATA/outputs/${SUB}_14/TwoStageModel/Run_${RUN_NUMBER}/logs"
    fi
    mkdir -p "$LOGDIR"

    echo "======================================"
    echo "▶ Running Stage 2 for $SUB."
    echo "Stage 1 model: $STAGE1_MODEL"
    echo "======================================"


        echo "▶ Tuning ($SUB)... Logfile: $LOGDIR/STAGE_2.out"
    python -u 49_2_TUNING.py "$SUB" "$RUN_NUMBER" "$COLLECTION" "$CALIBRATION" "$N_CLASSES" > "$LOGDIR/STAGE_2.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Tuning failed for $SUB. Skipping next steps."
        continue
    fi

    echo "▶ Hyperparameter tuning ($SUB)..."
    python -u 49_3_HYPERPARAMETER_TUNING.py "$SUB" "$RUN_NUMBER" "$COLLECTION" > "$LOGDIR/STAGE_3.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Hyperparameter tuning failed for $SUB. Skipping next step."
        continue
    fi

    echo "▶ Final model evaluation ($SUB)..."
    python -u 49_4_MODEL_balanced.py "$SUB" "$RUN_NUMBER" "$COLLECTION" > "$LOGDIR/STAGE_4.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Final model failed for $SUB."
        continue
    fi

    echo "✅ $SUB pipeline completed successfully."

done

echo "✅ All pipelines completed."


