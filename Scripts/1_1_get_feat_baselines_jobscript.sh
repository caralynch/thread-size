#!/bin/bash -l
set -euo pipefail
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
    echo "Usage: qsub 1_1_get_feat_baselines_jobscript.sh <subreddit>"
    exit 1
fi

# ===== Activate environment =====
source ~/.bashrc
conda activate threadsize 

# ===== Set paths =====
SUBREDDIT=$1  # Change as needed: conspiracy, crypto, politics
SCRIPT_DIR=/home/ucabcpl/Scratch/thread_size/thread-size/Scripts/1_Thread_start # Adjust to where your scripts are
DATA_DIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Preprocessing/${SUBREDDIT}" # Adjust to your data location

# Input files (adjust paths to your actual data)
TRAIN_X="$DATA_DIR/${SUBREDDIT}_train_X.parquet"
TRAIN_Y="$DATA_DIR/${SUBREDDIT}_train_Y.parquet"

# Output directory
OUTDIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Thread_Size/${SUBREDDIT}/1_feat_baselines"
mkdir $OUTDIR

# ===== Verify inputs exist =====
if [ ! -f "$TRAIN_X" ]; then
  echo "ERROR: X train file not found: $TRAIN_X"
  exit 1
fi

if [ ! -f "$TRAIN_Y" ]; then
  echo "ERROR: Y train not found: $TRAIN_Y"
  exit 1
fi

# Set max feats
MAX_FEATS=50

# ===== Run script =====

echo "[INFO] Running thread size 1 feature baselines for subreddit: ${SUBREDDIT}"
echo "[INFO] Output directory: ${OUTDIR}"
mkdir -p "${OUTDIR}"

CMD="python "${SCRIPT_DIR}/1_feature_baselines.py" \
  --subreddit ${SUBREDDIT} \
  --outdir ${OUTDIR} \
  --train_X ${TRAIN_X} \
  --train_y ${TRAIN_Y} \
  --feats ${MAX_FEATS}"

echo "[INFO] Command:"
echo "${CMD}"
echo "===================================="

# Actually run it
eval "${CMD}"
EXIT_CODE=$?

echo "===================================="
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "===================================="

exit ${EXIT_CODE}