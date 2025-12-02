#!/bin/bash

# List of subreddits to process
SUBREDDITS=("conspiracy" "crypto" "politics")

JOBSCRIPT="/home/ucabcpl/Scratch/thread_size/thread-size/Jobscripts/2_thread_size_jobscript.sh"

# Make sure the jobscript exists
if [ ! -f "$JOBSCRIPT" ]; then
    echo "ERROR: Jobscript not found: $JOBSCRIPT"
    exit 1
fi

for SUB in "${SUBREDDITS[@]}"; do
    echo "Submitting thread size job for: $SUB"

    # Submit job AND pass subreddit as the first argument
    qsub -N ${SUB}_2 -o "${SUB}_2.out" -e "${SUB}_2.err" "$JOBSCRIPT" "$SUB"

    echo ""
done

echo "All thread start jobs submitted."