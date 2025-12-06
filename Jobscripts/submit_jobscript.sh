#!/bin/bash

# List of subreddits to process
SUBREDDITS=("conspiracy" "crypto" "politics")

JOBSCRIPT=$1
SUFFIX=$2

# Make sure the jobscript exists
if [ ! -f "$JOBSCRIPT" ]; then
    echo "ERROR: Jobscript not found: $JOBSCRIPT"
    exit 1
fi

for SUB in "${SUBREDDITS[@]}"; do
    echo "Submitting thread start job for: $SUB"

    # Submit job AND pass subreddit as the first argument
    qsub -N ${SUB:0:3}_${SUFFIX} -o "${SUB:0:3}_${SUFFIX}.out" -e "${SUB:0:3}_${SUFFIX}.err" "$JOBSCRIPT" "$SUB"

    echo ""
done

echo "All thread start jobs submitted."