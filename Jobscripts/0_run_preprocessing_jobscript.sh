#!/bin/bash -l
# ===== SGE resource requests =====
#$ -l h_rt=24:00:00           # wallclock limit (24 hours - adjust based on data size)
#$ -l mem=8G                   # RAM per core (8G x 8 cores = 64G total)
#$ -l tmpfs=400G               # fast local scratch for intermediate files
#$ -pe smp 8                   # 8 CPU cores for parallel processing
#$ -N reddit_preprocess        # job name
#$ -cwd                        # start in submit dir
# #$ -m be                     # (optional) email at begin/end
# #$ -M ucabcpl@ucl.ac.uk 

# ===== Job info =====
echo "===================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "TMPDIR: $TMPDIR"
echo "===================================="


if [ -z "$1" ]; then
    echo "ERROR: No subreddit supplied."
    echo "Usage: qsub 0_run_preprocessing_jobscript.sh <subreddit>"
    exit 1
fi

# ===== Activate environment =====
source ~/.bashrc
conda activate threadsize 

# ===== Set paths =====
SUBREDDIT=$1  # Change as needed: conspiracy, crypto, politics
SCRIPT_DIR=/home/ucabcpl/Scratch/thread_size/thread-size/Scripts # Adjust to where your scripts are
DATA_DIR=/home/ucabcpl/Scratch/thread_size/thread-size/Inputs # Adjust to your data location

# Input files (adjust paths to your actual data)
COMMENTS_IN="$DATA_DIR/${SUBREDDIT}_comments.parquet"
THREADS_IN="$DATA_DIR/${SUBREDDIT}_threads.parquet"

# Output directory - use TMPDIR for speed, then copy back
OUTDIR="/home/ucabcpl/Scratch/thread_size/thread-size/0_Preprocessing/${SUBREDDIT}"
mkdir $OUTDIR

# ===== Verify inputs exist =====
if [ ! -f "$COMMENTS_IN" ]; then
  echo "ERROR: Comments file not found: $COMMENTS_IN"
  exit 1
fi

if [ ! -f "$THREADS_IN" ]; then
  echo "ERROR: Threads file not found: $THREADS_IN"
  exit 1
fi

# ===== Run preprocessing =====
echo ""
echo "Starting preprocessing pipeline..."
echo "Subreddit: $SUBREDDIT"
echo "Comments: $COMMENTS_IN"
echo "Threads: $THREADS_IN"
echo "Output: $OUTDIR"
echo ""

bash "$SCRIPT_DIR/Jobscripts/0_run_preprocessing.sh" \
  "$SUBREDDIT" \
  "$OUTDIR" \
  "$COMMENTS_IN" \
  "$THREADS_IN"

EXIT_CODE=$?

# ===== Copy results back to permanent storage =====
if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "Preprocessing completed successfully"
  
else
  echo ""
  echo "ERROR: Preprocessing failed with exit code $EXIT_CODE"
  echo "Check logs in: $OUTDIR/logs/"
  
  exit $EXIT_CODE
fi

echo "===================================="
echo "Job finished: $(date)"
echo "===================================="