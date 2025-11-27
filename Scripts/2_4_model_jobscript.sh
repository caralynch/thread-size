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
TEST_X="${DATA_DIR}/${SUBREDDIT}_test_X.parquet"
TEST_Y="${DATA_DIR}/${SUBREDDIT}_test_Y.parquet"
TUNING_PARAMS="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Thread_Size/${SUBREDDIT}/3_tuning/params_post_hyperparam_tuning.jl"
TFIDF_MODEL="${DATA_DIR}/tf-idf/${SUBREDDIT}_optuna_tfidf_vectorizer.jl"
SVD_MODEL="${DATA_DIR}/tf-idf/${SUBREDDIT}_optuna_svd_model.jl"


# Output directory
OUTDIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Thread_Size/${SUBREDDIT}/4_model"
mkdir -p "${OUTDIR}"

LOGDIR="${OUTDIR}/logs"
mkdir -p "${LOGDIR}"

# ===== Verify inputs exist =====
echo "[INFO] Checking required input files..."
for f in "${TRAIN_X}" "${TEST_X}" "${TRAIN_Y}" "${TEST_Y}" "${TUNING_PARAMS}" "${TFIDF_MODEL}" "${SVD_MODEL}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

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

echo "[INFO] Running Stage 2.4 tuned model for subreddit: ${SUBREDDIT}"
echo "[INFO] Output directory: ${OUTDIR}"

run_step "${SUBREDDIT}_4_model" \
  "${SCRIPT_DIR}/4_run_tuned_model.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${OUTDIR}" \
  --train_X "${TRAIN_X}" \
  --train_y "${TRAIN_Y}" \
  --test_X "${TEST_X}" \
  --test_y "${TEST_Y}" \
  --params "${TUNING_PARAMS}" \
  --tfidf "${TFIDF_MODEL}" \
  --svd "${SVD_MODEL}"


EXIT_CODE=$?

echo "===================================="
echo "Job finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "===================================="

exit "${EXIT_CODE}"
