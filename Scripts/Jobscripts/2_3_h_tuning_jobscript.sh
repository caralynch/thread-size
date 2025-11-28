#!/bin/bash -l
# ===== SGE resource requests =====
#$ -l h_rt=24:00:00           # wallclock limit (24 hours - adjust based on data size)
#$ -l mem=8G                  # RAM per core (8G x 8 cores = 64G total)
#$ -l tmpfs=400G              # fast local scratch for intermediate files
#$ -pe smp 8                  # 8 CPU cores for parallel processing
#$ -N threadsize_feat_baselines   # job name
#$ -cwd                       # start in submit dir
# #$ -m be                     # (optional) email at begin/end
# #$ -M ucabcpl@ucl.ac.uk 

# ===== Job info =====
echo "===================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "TMPDIR: ${TMPDIR:-/tmp}"
echo "===================================="

if [ -z "${1-}" ]; then
    echo "ERROR: No subreddit supplied."
    echo "Usage: qsub 2_3_h_tuning_jobscript.sh <subreddit>"
    exit 1
fi

# ===== Activate environment =====
source ~/.bashrc
conda activate threadsize 

# ===== Set paths =====
SUBREDDIT="$1"  # e.g. conspiracy, crypto, politics
SCRIPT_DIR="/home/ucabcpl/Scratch/thread_size/thread-size/Scripts/2_Thread_size"
DATA_DIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Preprocessing/${SUBREDDIT}"

# Input files
TRAIN_X="${DATA_DIR}/${SUBREDDIT}_train_X.parquet"
TRAIN_Y="${DATA_DIR}/${SUBREDDIT}_train_Y.parquet"
TUNING_PARAMS="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Thread_Size/${SUBREDDIT}/2_tuning/tuned_params.jl"

# Output directory
OUTDIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Thread_Size/${SUBREDDIT}/3_hyperparameter_tuning"
mkdir -p "${OUTDIR}"

LOGDIR="${OUTDIR}/logs"
mkdir -p "${LOGDIR}"

# ===== Verify inputs exist =====
if [ ! -f "${TRAIN_X}" ]; then
  echo "ERROR: X train file not found: ${TRAIN_X}"
  exit 1
fi

if [ ! -f "${TRAIN_Y}" ]; then
  echo "ERROR: Y train file not found: ${TRAIN_Y}"
  exit 1
fi

run_step () {
  local name="$1"; shift
  local log="${LOGDIR}/${name}.out"
  echo "== ${name} =="
  echo "Logging to ${log}"
  if ! python -u "$@" > "${log}" 2>&1; then
    echo "${name} failed. See ${log}"
    exit 1
  fi
}

# ===== Run script =====

echo "[INFO] Running Stage 2.1 feature baselines for subreddit: ${SUBREDDIT}"
echo "[INFO] Output directory: ${OUTDIR}"

run_step "${SUBREDDIT}_3_h_tuning" \
  "${SCRIPT_DIR}/3_hyperparameter_tuning.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${OUTDIR}" \
  --train_X "${TRAIN_X}" \
  --train_y "${TRAIN_Y}" \
  --params "${TUNING_PARAMS}"

EXIT_CODE=$?

echo "===================================="
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "===================================="

exit "${EXIT_CODE}"
