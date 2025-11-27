#!/bin/bash

# List of subreddits to process
SUBREDDITS=("conspiracy" "crypto" "politics")

JOBSCRIPT="0_run_preprocessing_jobscript.sh"

# Make sure the jobscript exists
if [ ! -f "$JOBSCRIPT" ]; then
    echo "ERROR: Jobscript not found: $JOBSCRIPT"
    exit 1
fi

for SUB in "${SUBREDDITS[@]}"; do
    echo "Submitting preprocessing job for: $SUB"

    # Submit job AND pass subreddit as the first argument
    qsub -N ${SUB}_0 "$JOBSCRIPT" "$SUB"

    echo ""
done

echo "All preprocessing jobs submitted."