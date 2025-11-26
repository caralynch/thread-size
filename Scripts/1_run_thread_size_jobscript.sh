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

set -euo pipefail
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
    echo "Usage: qsub 1_run_thread_size_jobscript.sh <subreddit>"
    exit 1
fi

# ===== Activate environment =====
conda activate threadsize 

# ===== Set paths =====
SUBREDDIT=$1  # Change as needed: conspiracy, crypto, politics
SCRIPT_DIR=/home/ucabcpl/Scratch/thread_size/thread-size/Scripts/1_Thread_start # Adjust to where your scripts are
DATA_DIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Preprocessing/${SUBREDDIT}" # Adjust to your data location

# Input files (adjust paths to your actual data)
TRAIN_X="$DATA_DIR/${SUBREDDIT}_train_X.parquet"
TRAIN_Y="$DATA_DIR/${SUBREDDIT}_train_Y.parquet"
TEST_X="$DATA_DIR/${SUBREDDIT}_test_X.parquet"
TEST_Y="$DATA_DIR/${SUBREDDIT}_test_Y.parquet"

# ===== Verify inputs exist =====
echo "[INFO] Checking required input files..."
for f in "${TRAIN_X}" "${TEST_X}" "${TRAIN_Y}" "${TEST_Y}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# Main output directory
OUTDIR="/home/ucabcpl/Scratch/thread_size/thread-size/Outputs/Thread_Size/${SUBREDDIT}"
mkdir -p $OUTDIR

LOGDIR="${OUTDIR}/logs"
mkdir -p $LOGDIR

MAX_FEATS=25

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

TUNING_OUTDIR="${OUTDIR}/2_tuning"
echo "===================================="
echo "[INFO] Stage 1.2 – feature + weight + threshold tuning"
echo "[INFO] Output directory: ${TUNING_OUTDIR}"
echo "===================================="


echo "[INFO] Running thread size tuning for subreddit: ${SUBREDDIT}"
echo "[INFO] Output directory: ${TUNING_OUTDIR}"
mkdir -p "${TUNING_OUTDIR}"


# Actually run it
run_step "${SUBREDDIT}_tuning" "${SCRIPT_DIR}/2_tuning.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${TUNING_OUTDIR}" \
  --train_X "${TRAIN_X}" \
  --train_y "${TRAIN_Y}" \
  --feats "${MAX_FEATS}"

TUNED_PARAMS="${TUNING_OUTDIR}/tuned_params.jl"
if [ ! -f "${TUNED_PARAMS}" ]; then
    echo "ERROR: Expected tuned params not found: ${TUNED_PARAMS}"
    exit 1
fi
echo "[OK] Found tuned params: ${TUNED_PARAMS}"

# ====== Hyperparam tuning ============

HPT_OUTDIR="${OUTDIR}/3_h_tuning"

echo "===================================="
echo "[INFO] Stage 1.3 – tree hyperparameter tuning"
echo "[INFO] Output directory: ${HPT_OUTDIR}"
echo "===================================="

run_step "${SUBREDDIT}_hyperparam_tuning" "${SCRIPT_DIR}/3_hyperparameter_tuning.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${HPT_OUTDIR}" \
  --train_X "${TRAIN_X}" \
  --train_y "${TRAIN_Y}" \
  --params "${TUNED_PARAMS}"

HPARAM_PARAMS="${HPT_OUTDIR}/params_post_hyperparam_tuning.jl"
if [ ! -f "${HPARAM_PARAMS}" ]; then
    echo "ERROR: Expected post-hyperparam params not found: ${HPARAM_PARAMS}"
    exit 1
fi
echo "[OK] Found hyperparameter-tuned params: ${HPARAM_PARAMS}"

MODEL_OUTDIR="${OUTDIR}/4_model"

echo "===================================="
echo "[INFO] Stage 1.4 – final model training and evaluation"
echo "[INFO] Output directory: ${MODEL_OUTDIR}"
echo "===================================="


echo "[INFO] Checking required input files... (TF-IDF and SVD)"
# TF-IDF + SVD models (from 2_tf_idf_analysis.py)
TFIDF_MODEL="${DATA_DIR}/${SUBREDDIT}/tf-idf/${SUBREDDIT}_optuna_tfidf_vectorizer.jl"
SVD_MODEL="${DATA_DIR}/${SUBREDDIT}/tf-idf/${SUBREDDIT}_optuna_svd_model.jl"



for f in "${TFIDF_MODEL}" "${SVD_MODEL}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

echo "[OK] All required input files present."
  
run_step "${SUBREDDIT}_model" "${SCRIPT_DIR}/4_run_tuned_model.py" \
  --subreddit "${SUBREDDIT}" \
  --outdir "${HPT_OUTDIR}" \
  --train_X "${TRAIN_X}" \
  --test_X "${TEST_X}" \
  --train_y "${TRAIN_Y}" \
  --test_y "${TEST_Y}" \
  --params "${HPARAM_PARAMS}" \
  --tfidf "${TFIDF_MODEL}" \
  --svd "${SVD_MODEL}"

echo "[INFO] Stage 1.4 command:"
echo "  ${CMD_MODEL}"
eval "${CMD_MODEL}"

echo "===================================="
echo "Job finished: $(date)"
echo "===================================="


exit ${EXIT_CODE}