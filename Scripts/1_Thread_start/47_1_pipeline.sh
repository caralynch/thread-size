#!/bin/bash

# Usage: bash run_all_models.sh <run_number>
if [ $# -lt 3 ]; then
    echo "Usage: bash run_all_models.sh <run_number> <collection> <calibration> <scoring_metric>"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: $0 $@"

RUN_NUMBER=$1
COLLECTION=$2
CALIBRATION=$3
METRIC=$4
SUBREDDITS=("crypto" "conspiracy" "politics")

for SUB in "${SUBREDDITS[@]}"; do
    START=$(date +%s)
    echo "======================================"
    echo "▶ Running Stage 1 pipeline for $SUB"
    echo "======================================"

    if [ "$COLLECTION" -eq 0 ]; then
        LOGDIR="REGDATA/outputs/${SUB}/Run_${RUN_NUMBER}/logs"
    else
        LOGDIR="REGDATA/outputs/${SUB}_14/TwoStageModel/Run_${RUN_NUMBER}/logs"
    fi
    mkdir -p "$LOGDIR"

    # echo "▶ Tuning ($SUB)..."
    # python -u 47_STAGE_1_2_TUNING.py "$SUB" "$RUN_NUMBER" "$COLLECTION" "$CALIBRATION" "$METRIC" > "$LOGDIR/stage_1_2.out" 2>&1
    # if [ $? -ne 0 ]; then
    #     echo "❌ Tuning failed for $SUB. Skipping next steps."
    #     continue
    # fi

    # echo "▶ Hyperparameter tuning ($SUB)..."
    # python -u 47_STAGE_1_3_HYPERPARAMETER_TUNING.py "$SUB" "$RUN_NUMBER" "$COLLECTION" > "$LOGDIR/stage_1_3.out" 2>&1
    # if [ $? -ne 0 ]; then
    #     echo "❌ Hyperparameter tuning failed for $SUB. Skipping next step."
    #     continue
    # fi

    echo "▶ Final model evaluation ($SUB)..."
    python -u 47_STAGE_1_4_MODEL.py "$SUB" "$RUN_NUMBER" "$COLLECTION" > "$LOGDIR/stage_1_4.out" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Final model failed for $SUB."
        continue
    fi

    echo "✅ $SUB pipeline part 1 completed successfully."
    END=$(date +%s)
    echo "⏱️ Duration: $((END - START)) seconds"
    echo ""
done

echo "✅ All pipelines completed."
