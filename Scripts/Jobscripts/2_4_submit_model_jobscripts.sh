#!/bin/bash

# List of subreddits to process
SUBREDDITS=("conspiracy" "crypto" "politics")

JOBSCRIPT="2_4_model_jobscript.sh"

# Make sure the jobscript exists
if [ ! -f "$JOBSCRIPT" ]; then
    echo "ERROR: Jobscript not found: $JOBSCRIPT"
    exit 1
fi

for SUB in "${SUBREDDITS[@]}"; do
    echo "Submitting thread size feature baselines job for: $SUB"

    # Submit job AND pass subreddit as the first argument
    qsub -N ${SUB}_2_4 -o "${SUB}_2_4.out" -e "${SUB}_2_4.err" "$JOBSCRIPT" "$SUB"

    echo ""
done

echo "All preprocessing jobs submitted."